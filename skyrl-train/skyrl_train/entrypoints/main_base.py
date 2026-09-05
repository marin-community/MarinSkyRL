"""
Main entrypoint for training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ray.util.placement_group import placement_group, PlacementGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.remote_function import RemoteFunction

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import ray

import os
import signal
import time
import hydra
from loguru import logger
import asyncio
import multiprocessing as mp

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.trajectory_runners.base import TrajectoryRunner

# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]


def resolve_entrypoint_node_id(node_ip: str) -> str:
    """Return the live Ray node ID for an explicitly selected node address."""
    matching_node_ids = [
        node["NodeID"] for node in ray.nodes() if node.get("Alive") and node.get("NodeManagerAddress") == node_ip
    ]
    if len(matching_node_ids) != 1:
        raise ValueError(
            f"Expected exactly one live Ray node with address {node_ip!r}, found {len(matching_node_ids)}. "
            "Set trainer.entrypoint_node_ip to a NodeManagerAddress reported by ray.nodes()."
        )
    return matching_node_ids[0]


class EntrypointSupervisor:
    """Wait for an entrypoint task and cooperatively cancel it on termination."""

    def __init__(self, *, shutdown_timeout_seconds: float) -> None:
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._termination_signal: int | None = None

    def request_termination(self, signum: int, _frame=None) -> None:
        """Record the first termination signal without calling Ray from a signal handler."""
        if self._termination_signal is None:
            self._termination_signal = signum

    def wait(self, entrypoint_ref: ray.ObjectRef) -> int | None:
        """Wait for the task, returning ``None`` normally or ``128 + signal`` after termination."""
        while self._termination_signal is None:
            ready, _ = ray.wait([entrypoint_ref], timeout=1)
            if ready:
                ray.get(entrypoint_ref)
                return None

        signum = self._termination_signal
        logger.warning(
            "Received {}, asking the training entrypoint to shut down within {}s",
            signal.Signals(signum).name,
            self._shutdown_timeout_seconds,
        )
        ray.cancel(entrypoint_ref, force=False, recursive=False)
        deadline = time.monotonic() + self._shutdown_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.error("Training entrypoint did not stop before its shutdown deadline; forcing cancellation")
                ray.cancel(entrypoint_ref, force=True, recursive=False)
                break
            ready, _ = ray.wait([entrypoint_ref], timeout=min(1, remaining))
            if ready:
                try:
                    ray.get(entrypoint_ref)
                except Exception as e:
                    # Cooperative task cancellation is reported as a Ray task
                    # error after the entrypoint's Python finally blocks finish.
                    logger.info("Training entrypoint stopped after the termination request: {}", e)
                break
        return 128 + signum


def create_ray_wrapped_inference_engines_from_config(cfg: DictConfig, colocate_pg, tokenizer: PreTrainedTokenizerBase):
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines

    engine_kwargs = {
        "num_inference_engines": cfg.generator.num_inference_engines,
        "tensor_parallel_size": cfg.generator.inference_engine_tensor_parallel_size,
        "pipeline_parallel_size": cfg.generator.inference_engine_pipeline_parallel_size,
        "model_dtype": cfg.generator.model_dtype,
        "pretrain": cfg.trainer.policy.model.path,
        "seed": cfg.trainer.seed,
        "vllm_v1_disable_multiproc": cfg.generator.vllm_v1_disable_multiproc,
        "enable_prefix_caching": cfg.generator.enable_prefix_caching,
        "enforce_eager": cfg.generator.enforce_eager,
        "expert_parallel_size": cfg.generator.inference_engine_expert_parallel_size,
        "data_parallel_size": cfg.generator.inference_engine_data_parallel_size,
        # vLLM Decode Context Parallel (DCP). Default 1 (disabled) -> forwarded as the
        # signature default and (per ray_wrapped_inference_engine) NOT passed to the vLLM
        # engine, so flag-off engine init is byte-identical to today (G1). When > 1 it is
        # threaded into vllm.LLM / AsyncEngineArgs as a native EngineArgs kwarg. DCP rides
        # the TP GPUs and is NOT part of any GPU/placement math (G4). Reaches both the
        # standard and terminal_bench entrypoints via this shared config-assembly seam (G5).
        "decode_context_parallel_size": cfg.generator.get("inference_engine_decode_context_parallel_size", 1),
        "shared_pg": colocate_pg,
        "engine_init_timeout_seconds": cfg.generator.engine_init_timeout_seconds,
        "gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
        "inference_engine_enable_sleep": cfg.trainer.placement.colocate_all,
        "async_engine": cfg.generator.async_engine,
        "max_num_batched_tokens": cfg.generator.max_num_batched_tokens,
        "max_num_seqs": cfg.generator.max_num_seqs,
        "tokenizer": tokenizer,
        "backend": cfg.generator.backend,
        "vllm_attention_backend": cfg.generator.get("vllm_attention_backend", None),
        "engine_init_kwargs": {
            **OmegaConf.to_container(cfg.generator.engine_init_kwargs, resolve=True),
            "openai_sampling_params": OmegaConf.to_container(cfg.generator.sampling_params, resolve=True),
        },
        # Opt-in mp executor backend (Qwen3-Next R3 capture hang workaround; default off).
        "mp_backend": cfg.generator.get("inference_engine_mp_backend", False),
        "placement_group_timeout_seconds": int(cfg.trainer.distributed.placement_group_timeout_seconds),
    }

    # Conditionally add LoRA parameters if LoRA is enabled
    if cfg.trainer.policy.model.lora.rank > 0:
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = cfg.trainer.policy.model.lora.rank
        engine_kwargs["sleep_level"] = 1
        engine_kwargs["max_loras"] = 1
        engine_kwargs["fully_sharded_loras"] = cfg.generator.fully_sharded_loras

        # TODO(devpatel): Bandaid solution, replace this once we have a better solution for LoRA performance degradation on the vLLM side
        if cfg.generator.enforce_eager and cfg.generator.backend == "vllm":
            logger.warning(
                "LoRA is enabled but generator.enforce_eager=true. "
                "This combination causes significant performance degradation (2-3x slower generation). "
                "Automatically setting enforce_eager=false for better performance. "
            )
            engine_kwargs["enforce_eager"] = False

    if (rope_scaling := cfg.generator.get("rope_scaling", None)) is not None:
        engine_kwargs["rope_scaling"] = rope_scaling
    if (rope_theta := cfg.generator.get("rope_theta", None)) is not None:
        engine_kwargs["rope_theta"] = rope_theta

    return create_ray_wrapped_inference_engines(**engine_kwargs)


def create_teacher_inference_engines_from_config(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase):
    """Create vLLM inference engines for the teacher model (distillation).

    Unlike the student engines, teacher engines:
    - Use the teacher model path (not policy model path)
    - Set max_logprobs to top_k_logprobs (not 1)
    - Don't enable sleep mode (teacher doesn't share GPU with training)
    - Don't set up weight sync (teacher weights are static)

    Also loads the teacher's own tokenizer for cross-model distillation.

    Returns:
        Tuple of (engines, teacher_tokenizer).
    """
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines

    teacher_cfg = cfg.teacher

    # Load teacher's own tokenizer for cross-model retokenization.
    # The teacher vLLM engine uses its own tokenizer internally for vocab
    # validation, so we must send it token IDs in its own vocabulary.
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_cfg.model_path, trust_remote_code=True)
    logger.info(f"Loaded teacher tokenizer: {teacher_cfg.model_path} (vocab_size={teacher_tokenizer.vocab_size})")

    engine_kwargs = {
        "num_inference_engines": teacher_cfg.num_inference_engines,
        "tensor_parallel_size": teacher_cfg.inference_engine_tensor_parallel_size,
        "pipeline_parallel_size": teacher_cfg.inference_engine_pipeline_parallel_size,
        "model_dtype": "auto",
        "pretrain": teacher_cfg.model_path,
        "seed": cfg.trainer.seed,
        "vllm_v1_disable_multiproc": False,
        "enable_prefix_caching": False,
        "enforce_eager": teacher_cfg.enforce_eager,
        "engine_init_timeout_seconds": teacher_cfg.engine_init_timeout_seconds,
        "expert_parallel_size": 1,
        "data_parallel_size": 1,
        "shared_pg": None,  # teacher gets its own placement group
        "gpu_memory_utilization": teacher_cfg.gpu_memory_utilization,
        "inference_engine_enable_sleep": False,  # teacher doesn't share GPU
        "async_engine": False,
        "max_num_batched_tokens": None,
        "max_num_seqs": None,
        "tokenizer": teacher_tokenizer,
        "backend": teacher_cfg.backend,
        "engine_init_kwargs": {
            **OmegaConf.to_container(teacher_cfg.engine_init_kwargs, resolve=True),
        },
        "max_logprobs": teacher_cfg.top_k_logprobs,
    }

    engines = create_ray_wrapped_inference_engines(**engine_kwargs)
    return engines, teacher_tokenizer


def create_remote_inference_engines_from_config(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase):
    from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines  # noqa: PLC0415

    # TODO(tgriggs): We may want a separate config for the model name in case it's different from the name used in the OpenAI API
    return create_remote_inference_engines(
        urls=cfg.generator.remote_inference_engine_urls,
        model_name=cfg.trainer.policy.model.path,
        engine_backend=cfg.generator.backend,
        tokenizer=tokenizer,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
        # DCP metadata only: for remote engines the operator must pass `-dcp <n>` on the
        # external `vllm serve` launch — SkyRL does not spawn remote servers. Carried here
        # for geometry/GPU-accounting consistency (DCP reuses the TP GPUs; no extra GPUs).
        decode_context_parallel_size=cfg.generator.get("inference_engine_decode_context_parallel_size", 1),
    )


class BasePPOExp:
    def __init__(self, cfg: DictConfig):
        """
        Initializes a PPO experiment.

        The `cfg` passed here will be the final config from Hydra, including CLI overrides.
        """
        self.cfg = cfg
        self._configure_log_level()
        self.tokenizer = self.get_tokenizer()
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.get_eval_dataset()
        self.colocate_pg = self.get_colocate_pg()
        # Reserve the policy/training placement group BEFORE the inference
        # engines (which are created later, in `_setup_trainer`), so that in the
        # disaggregated no-ref case the policy claims its dedicated whole nodes
        # first and the inference engines are forced onto the disjoint
        # remainder. None unless `policy_strict_spread_pg` is enabled for an
        # eligible (disaggregated, no-ref) run.
        self.policy_pg = self.get_policy_pg()

    def create_inference_engine_client(self) -> InferenceEngineClient:
        """Create the configured local or remote inference-engine client."""
        from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient  # noqa: PLC0415

        engine_mode = "local" if self.cfg.generator.run_engines_locally else "remote"
        logger.info("Starting inference engines: mode={}", engine_mode)
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(
                self.cfg, self.colocate_pg, self.tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, self.tokenizer)
        logger.info("Inference engines ready: mode={} count={}", engine_mode, len(inference_engines))
        return InferenceEngineClient(inference_engines, self.tokenizer, self.cfg)

    def _configure_log_level(self):
        """Configure loguru log level from trainer config."""
        import sys

        log_level = getattr(self.cfg.trainer, "log_level", "INFO").upper()
        # Remove default handler and add one with configured level
        logger.remove()
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
            colorize=True,
        )
        logger.info(f"SkyRL log level set to: {log_level}")

    @staticmethod
    def get_cfg_as_str(dict_cfg: DictConfig) -> str:
        return OmegaConf.to_yaml(dict_cfg)

    def get_tokenizer(self, padding_side="left"):
        """Initializes a tokenizer for the given model."""
        from skyrl_train.tokenizer import create_tokenizer  # noqa: PLC0415

        return create_tokenizer(
            model_path=self.cfg.trainer.policy.model.path,
            disable_fast_tokenizer=self.cfg.trainer.disable_fast_tokenizer,
            padding_side=padding_side,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            PromptDataset: The training dataset.
        """
        from skyrl_train.dataset import PromptDataset  # noqa: PLC0415

        prompts_dataset = PromptDataset(
            datasets=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=8,
        )
        # make sure the dataset is large enough to train on
        assert len(prompts_dataset) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        )
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            PromptDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            from skyrl_train.dataset import PromptDataset  # noqa: PLC0415

            prompts_dataset = PromptDataset(
                datasets=self.cfg.data.val_data,
                tokenizer=self.tokenizer,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                num_workers=8,
            )
            return prompts_dataset
        return None

    def get_colocate_pg(self, timeout: int | None = None) -> PlacementGroup:
        """Initializes a placement group for colocated training.

        A single placement group that packs all the inference engines together is created.

        Args:
            timeout (int): The timeout for the placement group to be ready.

        Returns:
            PlacementGroup: The placement group for colocated training.
        """
        from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout  # noqa: PLC0415

        timeout = int(self.cfg.trainer.distributed.placement_group_timeout_seconds) if timeout is None else timeout
        if self.cfg.trainer.placement.colocate_all:
            pg = placement_group(
                [{"GPU": 1, "CPU": 1}]
                * self.cfg.generator.num_inference_engines
                * self.cfg.generator.inference_engine_tensor_parallel_size
                * self.cfg.generator.inference_engine_pipeline_parallel_size
                * self.cfg.generator.inference_engine_data_parallel_size,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            return pg
        else:
            return None

    def get_policy_pg(self, timeout: int | None = None):
        """Reserve a dedicated whole-node placement group for the policy.

        Uses STRICT_SPREAD so each policy node gets exactly one bundle holding
        all of that node's GPUs — guaranteeing the policy occupies a set of
        whole, dedicated nodes that the (PACK) inference-engine placement group
        cannot share. Returns None when not eligible (see
        `policy_strict_spread_eligible`), in which case the legacy lazy-PACK
        path in `PPORayActorGroup._initiate_actors` is used unchanged.

        When a ref model is present in the disaggregated path, policy and ref
        share a single placement group built inside `build_models`; that path
        is left entirely untouched (eligibility requires use_ref_model=False).
        """
        from skyrl_train.utils.utils import (
            get_ray_pg_ready_with_timeout,
            policy_per_gpu_bundles_enabled,
            policy_spread_bundles,
            policy_strict_spread_eligible,
        )  # noqa: PLC0415

        timeout = int(self.cfg.trainer.distributed.placement_group_timeout_seconds) if timeout is None else timeout
        if not policy_strict_spread_eligible(self.cfg):
            return None

        from ray.util.placement_group import placement_group as _placement_group

        bundles = policy_spread_bundles(self.cfg)
        per_gpu = policy_per_gpu_bundles_enabled(self.cfg)
        # Per-GPU bundles ({GPU:1} x world_size) must NOT use STRICT_SPREAD
        # (that would force each single-GPU bundle onto a distinct node).
        # PACK packs them ~num_gpus_per_node-per-node; reserving all world_size
        # GPU slots up front still forces the (PACK) inference engines onto the
        # disjoint remaining nodes — the disjointness guarantee comes from the
        # reservation ordering, not from the spread strategy. Whole-node bundles
        # keep STRICT_SPREAD (one whole-node bundle per node) as before.
        strategy = "PACK" if per_gpu else "STRICT_SPREAD"
        pg = _placement_group(bundles, strategy=strategy)
        get_ray_pg_ready_with_timeout(pg, timeout=timeout)
        logger.info(
            f"Reserved dedicated policy placement group (strategy={strategy}, "
            f"{'per-GPU {GPU:1}' if per_gpu else 'whole-node {GPU:N}'} bundles): "
            f"{self.cfg.trainer.placement.policy_num_nodes} node(s) x "
            f"{self.cfg.trainer.placement.policy_num_gpus_per_node} GPU, "
            f"reserved before inference-engine placement to guarantee disjoint nodes."
        )
        return pg

    def get_trajectory_runner(self, cfg, tokenizer, inference_engine_client):
        """Initialize the configured trajectory runner.

        Returns:
            TrajectoryRunner: The runner.
        """
        from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection  # noqa: PLC0415
        from skyrl_train.trajectory_runners.skyrl_gym import (  # noqa: PLC0415
            SkyRLGymTrajectoryRunner,
            TrajectoryPipeline,
        )
        from skyrl_train.trajectory_runners.step_wise import StepWiseRolloutCollector  # noqa: PLC0415

        pipeline = None
        if cfg.trainer.step_wise_training:
            pipeline = TrajectoryPipeline(
                StepWiseRolloutCollector,
                StepWiseTrajectoryProjection(cfg.generator, tokenizer),
            )
        return SkyRLGymTrajectoryRunner(
            trajectory_runner_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            pipeline=pipeline,
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        trajectory_runner: TrajectoryRunner,
        colocate_pg,
    ):
        """Initializes the trainer.

        Returns:
            RayPPOTrainer: The trainer.
        """
        from skyrl_train.trainer import RayPPOTrainer  # noqa: PLC0415

        return RayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            trajectory_runner=trajectory_runner,
            colocate_pg=colocate_pg,
        )

    def get_tracker(self):
        """Initializes the tracker for experiment tracking.

        Returns:
            Tracking: The tracker.
        """
        from skyrl_train.utils.tracking import Tracking  # noqa: PLC0415

        return Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )

    def _setup_trainer(self):
        """Setup and return the trainer.

        Instantiates the trainer and all the associated models for training.

        Returns:
            RayPPOTrainer: The trainer.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy == "deepspeed":
            from skyrl_train.workers.deepspeed.deepspeed_worker import (
                PolicyWorker,
                CriticWorker,
                RefWorker,
            )
        elif self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        # NOTE (sumanthrh): Instantiate tracker before trainer init.
        # We have custom validation before this step to give better error messages.
        tracker = self.get_tracker()
        self.tracker = tracker

        tokenizer = self.tokenizer
        inference_engine_client = self.create_inference_engine_client()

        trajectory_runner: TrajectoryRunner = self.get_trajectory_runner(self.cfg, tokenizer, inference_engine_client)

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            trajectory_runner=trajectory_runner,
            colocate_pg=self.colocate_pg,
        )

        # Build the models. Pass the pre-reserved dedicated policy placement
        # group (None unless `policy_strict_spread_pg` is enabled for an
        # eligible disaggregated no-ref run).
        logger.info("Starting policy workers: strategy={}", self.cfg.trainer.strategy)
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker, policy_pg=self.policy_pg)
        logger.info(
            "Policy workers ready: strategy={} count={}",
            self.cfg.trainer.strategy,
            len(trainer.policy_model.actor_infos),
        )
        return trainer

    def run(self):
        from skyrl_train.telemetry import TRAINER_ROLE, process_telemetry
        from skyrl_train.utils.progress import configure_progress  # noqa: PLC0415 - keep launcher imports Torch-free

        configure_progress(self.cfg.trainer.progress)

        with process_telemetry(TRAINER_ROLE):
            self._run()

    def _run(self):
        # Force the orchestrator onto CPython's stock asyncio event loop (epoll),
        # NOT uvloop. Ray installs uvloop globally in every worker by default
        # (RAY_USE_UVLOOP defaults True -> default_worker.py:221 try_install_uvloop).
        # libuv's epoll-ctl machinery SIGABRTs this orchestrator under Daytona
        # sandbox-teardown socket churn (uv__epoll_ctl_prep AND uv__io_poll asserts;
        # present across libuv 1.45-1.49+). Reset the policy in this shared
        # BasePPOExp._run() path, which every training entrypoint funnels through
        # (main_base.skyrl_entrypoint, skyrl_train.entrypoints.terminal_bench's
        # TerminalBenchExp(BasePPOExp) which does NOT override run(), etc.) -- and
        # it runs immediately before the asyncio.run() below creates the loop, so
        # both asyncio.run() calls build a stock SelectorEventLoop with no libuv
        # path. Placing it on the per-entrypoint skyrl_entrypoint wrapper is a
        # trap: there are 26+ such functions and terminal_bench uses its own, so
        # the fix must live on this shared _run() method. Orchestrator is
        # network-RTT-bound (vLLM/Daytona) so uvloop's throughput edge is moot.
        #
        # DEPRECATION NOTE: asyncio.set_event_loop_policy() emits a
        # DeprecationWarning on Python 3.12+ and the policy system is slated for
        # removal (~3.16). It works on our 3.12 runtime. To future-proof when the
        # policy API is removed, drop this line and instead pass an explicit loop
        # to the asyncio.run() calls below:
        #   asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)  # Py 3.12+ Runner
        # See agent_logs/2026-05-29_skyrl_uvloop_integration_and_robustness_research.md
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        trainer = None
        self.tracker = None
        exit_code = 1
        try:
            trainer = self._setup_trainer()
            # Start the training loop
            asyncio.run(trainer.train())
            exit_code = 0
        finally:
            # Clean up any resources that were created, even if _setup_trainer()
            # or train() failed.  When the skyrl_entrypoint actor dies (e.g. SIGABRT)
            # and Ray retries it, the original actor's sub-actors (policy, ref,
            # inference engines) may still hold GPUs.  Cleaning up here ensures
            # those resources are released before the process exits.
            if trainer is not None:
                try:
                    # train() owns normal teardown. This idempotent fallback covers
                    # failures around the event-loop boundary.
                    asyncio.run(trainer.shutdown())
                except Exception as e:
                    logger.warning(f"Error shutting down trainer: {e}")
            if self.tracker is not None:
                try:
                    self.tracker.finish(exit_code=exit_code)
                except Exception:
                    if exit_code == 0:
                        raise
                    logger.exception("Tracker cleanup failed while handling a training failure")


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # NOTE: the uvloop->stock-asyncio reset that prevents the libuv epoll SIGABRT
    # lives in BasePPOExp.run() (the shared chokepoint all entrypoints funnel
    # through), NOT here -- terminal_bench and other entrypoints use their own
    # skyrl_entrypoint wrappers, so the fix must be on run(). See run() above.
    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


def run_ray_driver(cfg: DictConfig, entrypoint: RemoteFunction, *, failure_message: str = "Training failed") -> None:
    """Run one packaged experiment entrypoint with the shared Ray driver lifecycle."""
    from skyrl_train.entrypoints.ray_lifecycle import exit_without_ray_destructors, shutdown_ray  # noqa: PLC0415
    from skyrl_train.telemetry import DRIVER_ROLE, process_telemetry  # noqa: PLC0415
    from skyrl_train.utils import validate_cfg  # noqa: PLC0415
    from skyrl_train.utils.logging_utils import log_exception_as_text  # noqa: PLC0415
    from skyrl_train.utils.progress import configure_progress  # noqa: PLC0415 - keep launcher imports Torch-free
    from skyrl_train.utils.utils import initialize_ray  # noqa: PLC0415

    validate_cfg(cfg)
    configure_progress(cfg.trainer.progress)

    initialize_ray(cfg)

    with process_telemetry(DRIVER_ROLE):
        supervisor = EntrypointSupervisor(
            shutdown_timeout_seconds=float(cfg.trainer.get("entrypoint_shutdown_timeout_seconds", 120))
        )
        signal.signal(signal.SIGTERM, supervisor.request_termination)

        entrypoint_node_ip = cfg.trainer.get("entrypoint_node_ip")
        if entrypoint_node_ip:
            node_id = resolve_entrypoint_node_id(str(entrypoint_node_ip))
            logger.info("Pinning training entrypoint to Ray node {} ({})", node_id, entrypoint_node_ip)
            entrypoint = entrypoint.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            )

        exit_code = None
        try:
            exit_code = supervisor.wait(entrypoint.remote(cfg))
        except Exception as e:
            log_exception_as_text(failure_message, e)
            raise
        finally:
            logger.info("Shutting down Ray on head node...")
            shutdown_ray()

        if exit_code is not None:
            exit_without_ray_destructors(exit_code)
            raise SystemExit(exit_code)

    exit_without_ray_destructors()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_ray_driver(cfg, skyrl_entrypoint)


if __name__ == "__main__":
    main()
