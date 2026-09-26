import asyncio
from dataclasses import replace
import io as stdlib_io
import json
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float
from pathlib import Path
import ray
from ray import ObjectRef
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import PlacementGroup, placement_group
from skyrl_train.utils.progress import tqdm
from transformers import AutoTokenizer
from collections import defaultdict, deque

import numpy as np
from skyrl_train.dataset import PromptDataset
from skyrl_train.utils.tracking import Tracking
from skyrl_train.training_batch import GLOBAL_LOSS_DENOM_METADATA_KEY, TrainingInputBatch, TrainingOutputBatch
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import TrainingContext, TrainingContextState
from skyrl_train.trajectory_selection import trajectory_selector_from_config
from skyrl_train.trajectory_runners.base import (
    TrajectoryBatch,
    TrajectoryRunner,
)
import copy
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
    scalar_reward_token_credit,
)
from skyrl_train.trajectory_runners.trajectory_retention import make_trajectory_sink
from skyrl_train.dataset.preprocess import (
    collate_response_token_channel,
    convert_prompts_responses_to_batch_tensors,
)
from skyrl_train.distillation import DISTILLATION_SCORED_TOKENS_METRIC, validate_distillation_attachment
from skyrl_train.distillation_adapters import AsyncRoutedTeacherScoreTicket, RoutedScoredDistillationBatch
from skyrl_train.distillation_runtime import DistillationRuntime
from skyrl_train.domain_gradient_balance import DomainGradientBalancer
from skyrl_train.io import io
from skyrl_train.utils import Timer, get_ray_pg_ready_with_timeout, get_system_memory_metrics
from skyrl_train.tensor_math import masked_mean
from skyrl_train.utils.policy_math import compute_approx_kl, normalize_advantages_dict
from skyrl_train.utils.kl_controllers import get_kl_controller, FixedKLController, AdaptiveKLController
from skyrl_train.utils.advantage_estimators import compute_advantages_and_returns
from skyrl_train.utils.loss_reduction import (
    GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION,
    compute_global_loss_denom,
)
from skyrl_train.distributed.dispatch import (
    ActorInfo,
    MeshRank,
    collect_actor_results,
    concatenate_outputs_after_mesh_dispatch,
)
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    OnlineEagleCaptureConfig,
    OnlineEagleUpdateResult,
)
from skyrl_train.draft_trainer import (
    DraftUpdateRequest,
    create_draft_trainer,
    read_latest_draft_checkpoint,
)
from skyrl_train.group_admission import (
    AdmissionProgressWatchdog,
    GroupAdvantageInvariant,
    assert_training_groups_eligible,
)
from marinskyrl.checkpoint_paths import (
    GLOBAL_STEP_PREFIX,
    LATEST_CHECKPOINT_FILE,
)
from marinskyrl.resource_locator import is_cloud_uri, join_resource_path
from marinskyrl.speculative_decoding import SpeculativeDecodingConfig, runai_model_uri
from skyrl_train.checkpoint_listing import extract_step_from_path
from skyrl_train.utils.trainer_utils import (
    async_step_metrics,
    consumed_stop_metrics,
    cleanup_old_checkpoints,
    run_on_each_node,
    get_node_ids,
    validate_consistency_for_latest_checkpoint,
    ResumeMode,
    build_eval_dataloader,
)
from skyrl_train.utils.utils import (
    configure_ray_worker_logging,
    moe_router_replay_requested,
    policy_per_gpu_bundles_enabled,
    policy_force_cvd_mask_enabled,
)

from skyrl_train.utils.algorithm_registry import policy_loss_requires_rollout_logprobs
from skyrl_train.evaluate import evaluate, evaluate_step_wise
from skyrl_train.callbacks.base import TrainerCallback, TrainerState, TrainerControl, CallbackHandler
from skyrl_train.callbacks.builtin import DefaultCallbackHandler, RefModelUpdateCallback
from skyrl_train.telemetry import (
    TRAINER_ROLE,
    ConsumedWork,
    critical_phase,
    record_consumed_work,
    record_event,
    record_policy_step,
    record_rollout_staleness,
    record_training_metrics,
)
from skyrl_train.rollout_observability import async_phase_window, monitor_event_loop_lag
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics
from skyrl_train.timing_observability import publish_startup_timings, publish_step_timings
from skyrl_train.hf_export import (
    protected_hf_export_steps,
    read_hf_export_request,
    write_hf_export_request,
)
from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY, policy_export_path
from marinskyrl.runtime_options import WeightSyncTransport
from skyrl_train.utils.logging_utils import log_exception_as_text
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync
from skyrl_train.hf_export_schema import (
    DEFAULT_HF_HUB_REVISION,
    DEFAULT_HF_UPLOAD_MODE,
    HFExportRequest,
    HFExportStatus,
    HFUploadMode,
    TRAINER_STATE_FILENAME,
)


@dataclass(frozen=True)
class CheckpointSnapshot:
    step: int
    upload_started_at: float
    dataloader_path: str
    dataloader_payload: bytes
    trainer_state_path: str
    trainer_state_payload: bytes
    marker_path: str


_MODEL_INITIALIZATION_TIMEOUT = 60 * 60


def _active_online_eagle_results(results: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item for engine_results in results for item in engine_results if item.get("active", False)]


def _poll_object_ref(ref: ObjectRef) -> tuple[bool, Any | None]:
    ready, _ = ray.wait([ref], timeout=0)
    return (False, None) if not ready else (True, ray.get(ready[0]))


def _policy_revision(step: int) -> str:
    return f"policy-step-{step}"


def _validated_distillation_tensors(
    trajectory_batch: TrajectoryBatch,
    response_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Validate optional teacher evidence and return its learner-only tensors."""
    evidence = trajectory_batch.get("teacher_evidence")
    distillation = trajectory_batch.get("distillation")
    if evidence is None and distillation is None:
        return {}
    if evidence is None or distillation is None:
        raise ValueError("trajectory batches must carry teacher_evidence and prepared distillation together")
    trajectory_ids = trajectory_batch.get("trajectory_ids")
    if trajectory_ids is None:
        raise ValueError("teacher evidence requires stable trajectory_ids")
    validate_distillation_attachment(
        evidence,
        distillation,
        trajectory_ids=tuple(trajectory_id.to_string() for trajectory_id in trajectory_ids),
        response_mask=response_mask.to(torch.bool),
    )
    return distillation.training_tensors()


def consumed_work(training_input: TrainingInputBatch) -> ConsumedWork:
    """Count the rows and tokens an optimizer step consumed, excluding data-parallel padding."""
    real_rows = training_input.batch_size - training_input.metadata.get("pad_size", 0)
    return ConsumedWork(
        sequences=real_rows,
        response_tokens=int(training_input["response_mask"][:real_rows].sum().item()),
        loss_tokens=int(training_input["loss_mask"][:real_rows].sum().item()),
    )


class RayPPOTrainer:
    """The rollout-buffer training loop.

    Rollout workers generate prompt groups into the ``TrainingContext``'s buffer while the trainer reads one
    selected batch per step, trains on it, syncs weights to inference, and publishes the new policy step.
    ``global_step`` is the step being trained: its batch was generated by policies at most
    ``max_staleness_steps`` steps older, and ``global_step - 1`` optimizer steps have completed before it. At
    staleness 0 no rollout is in flight once a batch is complete, so colocated engines can sleep during training.
    """

    # Set at startup when generator.weight_sync_transport is expert_block.
    _expert_block_sync: ExpertBlockSync | None = None

    _training_metrics_enabled: bool = False
    _rollout_spans_enabled: bool = False

    def __init__(
        self,
        cfg: DictConfig,
        tracker: Tracking,
        tokenizer: AutoTokenizer,
        train_dataset: Optional[PromptDataset],
        inference_engine_client: InferenceEngineClient,
        trajectory_runner: TrajectoryRunner,
        context: TrainingContext,
        colocate_pg: Optional[PlacementGroup] = None,
        eval_dataset: Optional[PromptDataset] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        if not cfg.generator.async_engine:
            raise ValueError("the rollout-buffer training loop requires generator.async_engine=true")
        if cfg.trainer.placement.colocate_all and context.config.max_staleness_steps != 0:
            raise ValueError("colocate_all requires trainer.rollout_buffer.max_staleness_steps=0")
        self.cfg = cfg
        self.context = context
        self._training_metrics_enabled: bool = cfg.trainer.training_metrics
        self._rollout_spans_enabled: bool = cfg.trainer.rollout_spans
        self.group_advantage_invariant = GroupAdvantageInvariant.from_config(
            cfg.trainer.algorithm.resolved_group_advantage
        )
        self.colocate_all = cfg.trainer.placement.colocate_all
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.inference_engine_client = inference_engine_client
        self.trajectory_runner = trajectory_runner
        self.trajectory_selector = trajectory_selector_from_config(cfg)
        self.trajectory_sink = make_trajectory_sink(cfg.generator, tokenizer)
        self.trajectory_runner.set_trajectory_sink(self.trajectory_sink)
        self.total_training_steps = None
        self._configure_training_schedule()

        self.eval_dataloader = build_eval_dataloader(self.cfg, eval_dataset) if eval_dataset is not None else None
        self.colocate_pg = colocate_pg

        self.resume_mode = ResumeMode(cfg.trainer.resume_mode)

        self.all_metrics = {}
        self.all_timings = {}
        self.all_startup_timings = {}
        self._checkpoint_save_failures = 0.0
        self._shutdown_complete = False
        self._restored_rollout_state: TrainingContextState | None = None
        self.global_step = 0
        self._last_saved_step: int | None = None
        self._pending_checkpoint_upload: tuple[asyncio.Task[tuple[float, float]], TrainerState] | None = None
        raw_speculative_decoding = cfg.generator.get("speculative_decoding")
        if raw_speculative_decoding is not None:
            raw_speculative_decoding = OmegaConf.to_container(raw_speculative_decoding, resolve=True)
        self.speculative_decoding = (
            None
            if raw_speculative_decoding is None
            else SpeculativeDecodingConfig.from_mapping(raw_speculative_decoding)
        )
        self._speculator_capture_active = False
        self._speculator_revision = (
            None if self.speculative_decoding is None else self.speculative_decoding.model.source_identity
        )
        self._sealed_speculator_capture_uri: str | None = None
        self._speculator_checkpoint_root = join_resource_path(self.cfg.trainer.ckpt_path, "drafts")
        self._draft_trainer = None
        self._draft_trainer_update_ref: ObjectRef | None = None
        self._draft_trainer_submitted_at: float | None = None
        self._speculator_refresh_task: asyncio.Task | None = None
        self._speculator_requested_revision: str | None = None
        self._speculator_update_failures = 0
        self._speculator_install_count = 0
        self._speculator_install_failures = 0
        self._speculator_checkpoint_poll_failures = 0

        # initialized in `build_models`
        self.policy_model: PPORayActorGroup = None
        self.critic_model: Optional[PPORayActorGroup] = None
        self.ref_model: Optional[PPORayActorGroup] = None
        # used for checkpoint cleanup
        self._node_ids: Optional[List[str]] = None

        self.group_admission_stall_timeout = cfg.trainer.algorithm.group_admission.stall_timeout
        self._step_time_history: deque[float] = deque(maxlen=5)
        self._distillation_runtime: DistillationRuntime | None = None
        self._distillation_tickets: dict[str, AsyncRoutedTeacherScoreTicket] = {}
        self._domain_balancer: DomainGradientBalancer | None = None
        self.distillation_scored_tokens_total = 0

        self.reward_kl_controller: Optional[Union[FixedKLController, AdaptiveKLController]] = None
        configure_ray_worker_logging()

        # Initialize callback system
        # If callbacks are provided, use them; otherwise create defaults from config
        if callbacks is not None:
            self.callback_handler = CallbackHandler(callbacks)
        else:
            self.callback_handler = DefaultCallbackHandler(cfg)

        # Trainer control object for callback coordination
        self._control = TrainerControl()

    def configure_distillation(self, runtime: DistillationRuntime) -> None:
        """Install admitted-group teacher scoring before the training loop starts."""
        if self._distillation_runtime is not None:
            raise RuntimeError("the distillation runtime is already configured")
        self._distillation_runtime = runtime
        self._domain_balancer = runtime.domain_balancer

    def _configure_training_schedule(self):
        """Count steps in batches of prompt groups; one pass over the dataset is one epoch."""
        batch_size = self.cfg.trainer.train_batch_size
        self.num_steps_per_epoch = len(self.train_dataset) // batch_size
        if self.num_steps_per_epoch == 0:
            raise ValueError(
                f"the training dataset has {len(self.train_dataset)} prompts, fewer than one batch of {batch_size}"
            )
        self.total_training_steps = self.num_steps_per_epoch * self.cfg.trainer.epochs
        max_steps = self.cfg.trainer.get("max_steps")
        if max_steps is not None and max_steps > 0:
            self.total_training_steps = min(self.total_training_steps, max_steps)
        logger.info(f"Steps per epoch: {self.num_steps_per_epoch}, total training steps: {self.total_training_steps}")

    def _create_trainer_state(self, epoch: int, *, active_step_duration: float | None = None) -> TrainerState:
        """
        Create a TrainerState object for the current training state.

        This is used to pass immutable state information to callbacks.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            TrainerState object with current training state
        """
        num_steps_per_epoch = self._num_steps_per_epoch()
        timings = dict(self.all_timings)
        if active_step_duration is not None:
            timings["step"] = active_step_duration
        return TrainerState(
            global_step=self.global_step,
            epoch=epoch,
            total_steps=self.total_training_steps,
            num_steps_per_epoch=num_steps_per_epoch,
            is_last_step=(self.global_step == self.total_training_steps),
            is_epoch_end=(self.global_step % num_steps_per_epoch == 0) if num_steps_per_epoch > 0 else False,
            metrics=dict(self.all_metrics),
            timings=timings,
        )

    def _num_steps_per_epoch(self) -> int:
        return self.num_steps_per_epoch

    def _get_ref_update_callback(self) -> Optional[RefModelUpdateCallback]:
        """Get the RefModelUpdateCallback if one exists in the callback handler."""
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, RefModelUpdateCallback):
                return callback
        return None

    @torch.no_grad()
    async def eval(self) -> Dict[str, float]:
        """
        Run generation and scoring on the evaluation dataset.

        The eval metrics are recorded after having finished training `self.global_step` steps.
        Metrics recorded in global_step 0 corresponds to evaluations before training.

        Returns:
            A dictionary of evaluation metrics.
        """
        if self.cfg.trainer.step_wise_training:
            eval_metrics = await evaluate_step_wise(
                eval_dataloader=self.eval_dataloader,
                trajectory_runner=self.trajectory_runner,
                cfg=self.cfg,
                global_step=self.global_step,
                tokenizer=self.tokenizer,
                trajectory_sink=self.trajectory_sink,
            )
        else:
            eval_metrics = await evaluate(
                eval_dataloader=self.eval_dataloader,
                trajectory_runner=self.trajectory_runner,
                cfg=self.cfg,
                global_step=self.global_step,
                tokenizer=self.tokenizer,
                trajectory_sink=self.trajectory_sink,
            )
        return eval_metrics

    # ------------------------------------------------------------------
    # Teardown helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _guarded_async(coro, *, timeout: float, label: str) -> None:
        """Await *coro* with a timeout, logging but never raising on failure."""
        try:
            await asyncio.wait_for(coro, timeout=timeout)
            logger.info(f"{label} complete")
        except asyncio.TimeoutError:
            logger.warning(f"{label} timed out after {timeout}s, proceeding with cleanup")
        except Exception as e:
            logger.warning(f"{label} error (non-fatal): {e}")

    @staticmethod
    def _guarded_sync(fn, *, label: str) -> None:
        """Call *fn()*, logging but never raising on failure."""
        try:
            fn()
            logger.info(f"{label} complete")
        except Exception as e:
            logger.warning(f"{label} error (non-fatal): {e}")

    def cleanup_ray_actors(self):
        """Public alias for :meth:`_kill_ray_actors` (used by entrypoints)."""
        return self._kill_ray_actors()

    def _kill_ray_actors(self):
        """Kill all managed Ray actors (models + inference engines).

        Terminates policy, critic, and ref-model actor groups as well as
        inference engine actors (vLLM / SGLang workers) that would otherwise
        keep running until the node dies.
        """
        for model_name, model in [
            ("policy_model", self.policy_model),
            ("critic_model", self.critic_model),
            ("ref_model", self.ref_model),
        ]:
            if model is not None:
                try:
                    logger.info(f"Killing {model_name} actors...")
                    model.kill_actors()
                except Exception as e:
                    logger.warning(f"Error killing {model_name} actors: {e}")

        if getattr(self, "_draft_trainer", None) is not None:
            try:
                ray.kill(self._draft_trainer, no_restart=True)
                logger.info("Killed DraftTrainer actor")
            except Exception as error:
                logger.warning("Error killing DraftTrainer actor: {}", error)
            self._draft_trainer = None

        # Kill inference engine actors.  These are not covered by the model
        # actor groups above.
        if self.inference_engine_client is not None:
            from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine

            n_killed = 0
            for engine in self.inference_engine_client.engines:
                if isinstance(engine, RayWrappedInferenceEngine):
                    try:
                        ray.kill(engine.inference_engine_actor, no_restart=True)
                        n_killed += 1
                    except Exception:
                        pass  # Actor may already be dead
            if n_killed:
                logger.info(f"Killed {n_killed} inference engine actor(s)")

    async def _teardown(self) -> None:
        """Best-effort cleanup after training ends (normal or abnormal).

        Each step uses a timeout so a blocked operation cannot prevent
        subsequent cleanup from running.  Errors are logged as warnings
        but never re-raised.

        Order matters:
        1. HTTP endpoint shutdown – cuts off the request path so in-flight
           Harbor trials get connection-refused instead of retrying against
           dead inference engines indefinitely.
        2. Trajectory runner shutdown – waits for QueueOrchestrator to drain (should
           be fast now that trials can't make new requests).
        3. Inference engine teardown – sends teardown RPC to each engine.
        4. Ray actor cleanup – force-kills remaining actors.
        """
        if self._expert_block_sync is not None:
            await self._guarded_async(
                self._expert_block_sync.close(),
                timeout=30,
                label="Expert-block weight sync shutdown",
            )
        if self._distillation_runtime is not None:
            await self._guarded_async(self._distillation_runtime.close(), timeout=30, label="Teacher oracle shutdown")
        if self.inference_engine_client is not None:
            self._guarded_sync(
                self.inference_engine_client.shutdown_http_endpoint,
                label="HTTP endpoint shutdown",
            )
        await self._guarded_async(
            self.trajectory_runner.shutdown(),
            timeout=60,
            label="Trajectory runner shutdown",
        )
        self._guarded_sync(self.trajectory_sink.close, label="Trajectory retention shutdown")
        self._draft_trainer_update_ref = None
        if self._speculator_refresh_task is not None:
            await self._guarded_async(
                self._speculator_refresh_task,
                timeout=30,
                label="Draft refresh completion",
            )
            self._speculator_refresh_task = None
        await self._guarded_async(
            self.inference_engine_client.teardown(),
            timeout=30,
            label="Inference engine teardown",
        )
        self._guarded_sync(self._kill_ray_actors, label="Ray actor cleanup")

        # Safety net: force-exit the process if it's still alive after a
        # generous grace period.  After this point, asyncio.run() will try to
        # cancel remaining tasks (_cancel_all_tasks).  If orphaned tasks are
        # stuck in retry loops (e.g. Harbor trials retrying against dead
        # inference engines), that cleanup hangs indefinitely.  The watchdog
        # ensures the process eventually terminates.
        self._start_exit_watchdog(timeout=120)

    async def shutdown(self) -> None:
        """Run trainer teardown once, including after partial startup."""
        if getattr(self, "_shutdown_complete", False):
            return
        try:
            await self.context.close()
            await self._drain_checkpoint_upload()
        finally:
            await self._teardown()
            self._shutdown_complete = True

    @staticmethod
    def _start_exit_watchdog(timeout: int = 120) -> None:
        """Start a daemon thread that force-exits the process after *timeout* seconds."""

        def _force_exit():
            logger.error(f"Process still alive {timeout}s after teardown — forcing exit to prevent zombie process")
            os._exit(1)

        t = threading.Timer(timeout, _force_exit)
        t.daemon = True
        t.start()

    async def train(self):
        """Run the rollout-buffer training loop, then release every resource it started."""
        loop_monitor = (
            asyncio.create_task(monitor_event_loop_lag(step_fn=lambda: self.global_step, mode=self.context.mode))
            if self._rollout_spans_enabled
            else None
        )
        try:
            if self._distillation_runtime is not None:
                await self._distillation_runtime.start()
            await self._startup_trajectory_runner()
            await self._train_loop()
        except Exception as e:
            log_exception_as_text(f"Train loop failed at global_step {self.global_step}", e)
            raise
        finally:
            if loop_monitor is not None:
                loop_monitor.cancel()
            await self.shutdown()

    async def _startup_trajectory_runner(self) -> None:
        """Initialize trajectory-runner resources before any rollout can begin."""
        implementation = type(self.trajectory_runner).__name__
        logger.info("Starting trajectory runner: implementation={}", implementation)
        try:
            await self.trajectory_runner.startup()
            logger.info("Trajectory runner ready: implementation={}", implementation)
        except Exception as e:
            logger.opt(depth=0).error("Trajectory runner startup failed: " + str(e))
            raise

    async def _handle_resume_at_max_steps(self) -> None:
        """Handle a run that resumed AT or PAST max_steps (already complete).

        Fires on_train_end callbacks (so a missing final checkpoint / HF export /
        HF upload still runs) and returns without executing another training step,
        so the process exits 0 (clean COMPLETED) instead of overshooting to gs N+1.
        """
        logger.info(
            f"Resumed at global_step {self.global_step} >= max training steps "
            f"({self.total_training_steps}); run is already COMPLETE. Skipping further "
            f"training and finalizing (export/upload if missing)."
        )
        await self._finalize_training(
            completed_step=self.global_step,
            epoch=max(self.cfg.trainer.epochs - 1, 0),
        )
        logger.info("Training already complete on resume — exiting cleanly.")

    async def _finalize_training(self, *, completed_step: int, epoch: int) -> None:
        """Run train-end callbacks, evaluation, and saves at the last completed optimizer step."""
        await self._drain_checkpoint_upload()
        self.global_step = completed_step
        final_state = self._create_trainer_state(epoch=epoch)
        self._control.reset()
        self._control = await self.callback_handler.call_event_async(
            "on_train_end", final_state, self._control, trainer=self
        )

        try:
            if self._control.should_evaluate and self.eval_dataset is not None:
                with Timer("eval", self.all_timings):
                    eval_metrics = await self.eval()
                    self._log_metrics_stdout(eval_metrics, step=self.global_step, kind="eval")
                    self.tracker.log(eval_metrics, step=self.global_step, commit=True)
                await self.callback_handler.call_event_async(
                    "on_evaluate", final_state, self._control, metrics=eval_metrics, trainer=self
                )
                self._control.should_evaluate = False
        finally:
            if self.colocate_all:
                await self.inference_engine_client.sleep()
                self.policy_model.backload_to_gpu()

            # The interval save may have just written this same step; do not write it twice.
            if self._control.should_save and self._last_saved_step != self.global_step:
                with Timer("save_checkpoints", self.all_timings):
                    rollout_state = await self._rollout_state()
                    snapshot = await asyncio.to_thread(self._snapshot_checkpoint, rollout_state)
                try:
                    await self.callback_handler.call_event_async("on_save", final_state, self._control, trainer=self)
                except BaseException:
                    await self._await_checkpoint_upload(snapshot)
                    raise
                self._start_checkpoint_upload(snapshot, final_state)
                if await self._drain_checkpoint_upload():
                    logger.info("Saved final checkpoint.")
            if self._control.should_save_hf_model:
                await asyncio.to_thread(self.handle_hf_export)

    async def _save_checkpoints_with_residency(self) -> CheckpointSnapshot:
        """Save a checkpoint, swapping colocated training and inference residency when needed."""
        rollout_state = await self._rollout_state()
        if not self.colocate_all:
            return await asyncio.to_thread(self._snapshot_checkpoint, rollout_state)

        await self.inference_engine_client.sleep()
        try:
            self.policy_model.backload_to_gpu(backload_optimizer=True, backload_model=True)
            return await asyncio.to_thread(self._snapshot_checkpoint, rollout_state)
        finally:
            await self._sync_policy_for_rollouts(reason="checkpoint_restore")

    def _record_checkpoint_save_failure(self, state: TrainerState) -> None:
        self._checkpoint_save_failures += 1.0
        self.all_metrics["trainer/checkpoint_save_failures"] = self._checkpoint_save_failures
        logger.opt(exception=True).error(
            f"Checkpoint save failed at global step {state.global_step}; continuing from the last complete checkpoint"
        )

    async def _save_intermediate_checkpoint(self, state: TrainerState) -> None:
        """Save one requested step checkpoint without terminating training on storage failure."""
        await self._drain_checkpoint_upload()
        try:
            with Timer("save_checkpoints", self.all_timings):
                snapshot = await self._save_checkpoints_with_residency()
        except OSError:
            self._record_checkpoint_save_failure(state)
            return
        except ray.exceptions.RayTaskError as error:
            if not isinstance(error.as_instanceof_cause(), OSError):
                raise
            self._record_checkpoint_save_failure(state)
            return

        try:
            await self.callback_handler.call_event_async("on_save", state, self._control, trainer=self)
        except BaseException:
            await self._await_checkpoint_upload(snapshot)
            raise
        self._start_checkpoint_upload(snapshot, state)

    def _start_checkpoint_upload(self, snapshot: CheckpointSnapshot, state: TrainerState) -> None:
        assert getattr(self, "_pending_checkpoint_upload", None) is None
        task = asyncio.create_task(self._finish_checkpoint_upload(snapshot, commit=True))
        self._pending_checkpoint_upload = (task, state)

    def _finish_checkpoint_upload_blocking(self, snapshot: CheckpointSnapshot, *, commit: bool) -> tuple[float, float]:
        cleanup_duration = 0.0
        actor_refs = self.policy_model.async_run_ray_method("pass_through", "wait_checkpoint_upload")
        if self.critic_model is not None:
            actor_refs.extend(self.critic_model.async_run_ray_method("pass_through", "wait_checkpoint_upload"))
        ray.get(actor_refs)
        if commit:
            io.write_bytes_atomic(snapshot.trainer_state_path, snapshot.trainer_state_payload)
            io.write_bytes_atomic(snapshot.dataloader_path, snapshot.dataloader_payload)
            io.write_bytes_atomic(snapshot.marker_path, str(snapshot.step).encode())
            self._last_saved_step = snapshot.step
            cleanup_started = time.monotonic()
            self._cleanup_old_checkpoints()
            cleanup_duration = time.monotonic() - cleanup_started
        return time.monotonic() - snapshot.upload_started_at, cleanup_duration

    async def _finish_checkpoint_upload(self, snapshot: CheckpointSnapshot, *, commit: bool) -> tuple[float, float]:
        return await asyncio.to_thread(self._finish_checkpoint_upload_blocking, snapshot, commit=commit)

    async def _await_checkpoint_upload(self, snapshot: CheckpointSnapshot) -> None:
        await self._finish_checkpoint_upload(snapshot, commit=False)

    async def _drain_checkpoint_upload(self) -> bool:
        pending = getattr(self, "_pending_checkpoint_upload", None)
        if pending is None:
            return True
        task, state = pending
        self._pending_checkpoint_upload = None
        try:
            duration, cleanup_duration = await task
        except OSError:
            self._record_checkpoint_save_failure(state)
            return False
        except ray.exceptions.RayTaskError as error:
            if not isinstance(error.as_instanceof_cause(), OSError):
                raise
            self._record_checkpoint_save_failure(state)
            return False
        self.all_timings["checkpoint_upload"] = self.all_timings.get("checkpoint_upload", 0.0) + duration
        self.all_timings["cleanup_old_checkpoints"] = (
            self.all_timings.get("cleanup_old_checkpoints", 0.0) + cleanup_duration
        )
        return True

    async def _run_step_end_callbacks(self, state: TrainerState) -> None:
        """Run callback-requested work that belongs to the current training step."""
        self._control.reset()
        self._control = await self.callback_handler.call_event_async("on_step_end", state, self._control, trainer=self)

        if self._control.should_save:
            await self._save_intermediate_checkpoint(state)
            self._control.should_save = False

        if self._control.should_save_hf_model:
            # HF export reads the committed checkpoint. When checkpoint and HF
            # export share a cadence, wait for the background shard uploads and
            # marker publication before asking the exporter to consume it.
            if await self._drain_checkpoint_upload():
                await asyncio.to_thread(self.handle_hf_export)
            self._control.should_save_hf_model = False

        if self._control.should_evaluate and self.eval_dataset is not None:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self.all_metrics.update(eval_metrics)
            await self.callback_handler.call_event_async(
                "on_evaluate", state, self._control, metrics=eval_metrics, trainer=self
            )
            self._control.should_evaluate = False

    async def _start_draft_trainer(self) -> None:
        """Start the independent one-GPU draft trainer when online updates are enabled."""
        config = self.speculative_decoding
        if config is None or config.training is None or self._draft_trainer is not None:
            return
        if not is_cloud_uri(self._speculator_checkpoint_root):
            raise ValueError("Online EAGLE training requires trainer.ckpt_path to use s3://, gs://, or gcs://")
        self._draft_trainer = create_draft_trainer(
            initial_model=config.model,
            checkpoint_root=self._speculator_checkpoint_root,
        )
        logger.info("DraftTrainer ready: {}", await self._draft_trainer.status.remote())
        await self._refresh_latest_speculator(wait=True)

    def _should_update_speculator(self, step: int) -> bool:
        config = self.speculative_decoding
        return (
            config is not None
            and config.training is not None
            and self._draft_trainer_update_ref is None
            and step % config.training.interval_steps == 0
        )

    def _speculator_capture_uri(self, step: int) -> str:
        return join_resource_path(self._speculator_checkpoint_root, "captures", f"step-{step}")

    async def _begin_speculator_capture(self, step: int) -> None:
        """Capture rollouts generated for training step ``step`` without making draft failures fatal."""
        await self._poll_speculator_lifecycle()
        if self._speculator_capture_active or not self._should_update_speculator(step):
            return
        assert self.speculative_decoding is not None
        training = self.speculative_decoding.training
        assert training is not None
        assert self._speculator_revision is not None
        capture_config = OnlineEagleCaptureConfig(
            step=step,
            max_tokens=training.max_tokens_per_update,
            max_window_tokens=training.max_window_tokens,
            target_revision=_policy_revision(step - 1),
            draft_revision=self._speculator_revision,
            reserved_gpu_memory_gib=training.reserved_gpu_memory_gib,
        )
        capture_uri = self._speculator_capture_uri(step)
        try:
            if await asyncio.to_thread(io.exists, capture_uri):
                await asyncio.to_thread(io.remove, capture_uri)
            results = await self.inference_engine_client.begin_online_eagle_capture(capture_config.to_mapping())
        except Exception as error:
            logger.warning("Online EAGLE capture skipped at step {}: {}", step, error)
            self.all_metrics["speculator/capture_failures"] = 1.0
            return
        active = _active_online_eagle_results(results)
        if not active:
            logger.warning("Online EAGLE capture had no active ranks at step {}", step)
            return
        self._speculator_capture_active = True
        logger.info(
            "Online EAGLE capture started: step={} ranks={} target_revision={} draft_revision={}",
            step,
            len(active),
            capture_config.target_revision,
            capture_config.draft_revision,
        )

    async def _seal_speculator_capture(self) -> None:
        """Publish the active capture and retain only its cloud URI."""
        if not self._speculator_capture_active:
            return
        capture_uri = self._speculator_capture_uri(self.global_step)
        self._speculator_capture_active = False
        try:
            manifests = await self.inference_engine_client.seal_online_eagle_capture(capture_uri)
        except Exception as error:
            logger.warning("Online EAGLE capture publication failed at step {}: {}", self.global_step, error)
            self.all_metrics["speculator/capture_failures"] = 1.0
            return
        active = _active_online_eagle_results(manifests)
        if not active:
            logger.warning("Online EAGLE capture published no rank manifests at step {}", self.global_step)
            self.all_metrics["speculator/capture_failures"] = 1.0
            return
        self._sealed_speculator_capture_uri = capture_uri
        self.all_metrics.update(
            {
                "speculator/sealed_rows": float(sum(item.get("captured_rows", 0) for item in active)),
                "speculator/sealed_windows": float(sum(len(item.get("windows", ())) for item in active)),
                "speculator/capture_dropped_windows": float(sum(item.get("dropped_windows", 0) for item in active)),
            }
        )
        logger.info(
            "Online EAGLE capture published: step={} ranks={} uri={}",
            self.global_step,
            len(active),
            capture_uri,
        )

    async def _start_speculator_update(self) -> None:
        """Hand the published capture URI to the independent DraftTrainer actor."""
        capture_uri = self._sealed_speculator_capture_uri
        if capture_uri is None or self._draft_trainer_update_ref is not None:
            return
        if self._draft_trainer is None or self.speculative_decoding is None:
            raise RuntimeError("Online EAGLE capture published before DraftTrainer started")
        training = self.speculative_decoding.training
        assert training is not None
        request = DraftUpdateRequest(
            step=self.global_step,
            capture_uri=capture_uri,
            target_revision=_policy_revision(self.global_step - 1),
            num_speculative_tokens=self.speculative_decoding.num_speculative_tokens,
            seed=int(self.cfg.trainer.seed),
            training=training,
        )
        self._draft_trainer_update_ref = self._draft_trainer.update.remote(request)
        self._draft_trainer_submitted_at = time.monotonic()
        self._sealed_speculator_capture_uri = None
        self.all_metrics["speculator/update_pending"] = 1.0
        logger.info(
            "DraftTrainer update submitted: step={} capture_uri={}",
            self.global_step,
            capture_uri,
        )

    async def _start_latest_speculator_refresh(self) -> None:
        if (
            self.speculative_decoding is None
            or self.speculative_decoding.training is None
            or self._speculator_refresh_task is not None
        ):
            return
        try:
            checkpoint = await asyncio.to_thread(
                read_latest_draft_checkpoint,
                self._speculator_checkpoint_root,
                source_identity=self.speculative_decoding.model.source_identity,
            )
        except Exception as error:
            self._speculator_checkpoint_poll_failures += 1
            self.all_metrics["speculator/checkpoint_poll_failures"] = float(self._speculator_checkpoint_poll_failures)
            logger.warning("Draft checkpoint poll failed: {}", error)
            return
        if (
            checkpoint is None
            or checkpoint.revision == self._speculator_revision
            or checkpoint.revision == self._speculator_requested_revision
        ):
            return
        self._speculator_requested_revision = checkpoint.revision
        self._speculator_refresh_task = asyncio.create_task(
            self.inference_engine_client.update_draft_weights(
                runai_model_uri(checkpoint.weights_uri),
            )
        )
        await asyncio.sleep(0)

    async def _finish_speculator_refresh(self) -> None:
        task = self._speculator_refresh_task
        if task is None or not task.done():
            return
        requested_revision = self._speculator_requested_revision
        self._speculator_refresh_task = None
        try:
            coverage = task.result()
        except Exception as error:
            logger.warning("Draft refresh failed for revision {}: {}", requested_revision, error)
            self._speculator_requested_revision = None
            self._speculator_install_failures += 1
            return
        successful = [entry for entry in coverage if entry.get("active", False)]
        failed = [entry for entry in coverage if not entry.get("active", False)]
        self._speculator_install_failures += len(failed)
        if len(successful) == len(coverage) and coverage:
            self._speculator_revision = requested_revision
            self._speculator_install_count += 1
            logger.info(
                "Draft revision refreshed: revision={} engines={}",
                requested_revision,
                len(successful),
            )
        else:
            self._speculator_requested_revision = None
            logger.warning(
                "Draft revision refresh was partial: revision={} successful_engines={} failed_engines={}",
                requested_revision,
                len(successful),
                len(failed),
            )
        self.all_metrics.update(
            {
                "speculator/install_count": float(self._speculator_install_count),
                "speculator/install_failures": float(self._speculator_install_failures),
                "speculator/install_successful_engines": float(len(successful)),
                "speculator/install_failed_engines": float(len(failed)),
            }
        )

    async def _refresh_latest_speculator(self, *, wait: bool = False) -> None:
        await self._finish_speculator_refresh()
        await self._start_latest_speculator_refresh()
        if wait and self._speculator_refresh_task is not None:
            await self._speculator_refresh_task
            await self._finish_speculator_refresh()

    async def _poll_speculator_lifecycle(self) -> None:
        """Poll independent draft work and start any newly published refresh."""
        if self.speculative_decoding is None or self.speculative_decoding.training is None:
            return
        ref = self._draft_trainer_update_ref
        if ref is not None:
            try:
                ready, payload = _poll_object_ref(ref)
            except Exception as error:
                self._draft_trainer_update_ref = None
                self._draft_trainer_submitted_at = None
                self._speculator_update_failures += 1
                self.all_metrics["speculator/update_pending"] = 0.0
                self.all_metrics["speculator/update_failures"] = float(self._speculator_update_failures)
                logger.warning("DraftTrainer actor update failed: {}", error)
                await self._refresh_latest_speculator()
                return
            if ready:
                self._draft_trainer_update_ref = None
                if not isinstance(payload, OnlineEagleUpdateResult):
                    self._draft_trainer_submitted_at = None
                    self._speculator_update_failures += 1
                    self.all_metrics["speculator/update_pending"] = 0.0
                    self.all_metrics["speculator/update_failures"] = float(self._speculator_update_failures)
                    logger.warning(
                        "DraftTrainer returned {}, expected OnlineEagleUpdateResult",
                        type(payload).__name__,
                    )
                    await self._refresh_latest_speculator()
                    return
                result = payload
                submitted_at = self._draft_trainer_submitted_at
                self._draft_trainer_submitted_at = None
                if submitted_at is not None:
                    self.all_metrics["speculator/update_wall_seconds"] = time.monotonic() - submitted_at
                self.all_metrics.update(
                    {
                        "speculator/update_pending": 0.0,
                        "speculator/candidate_accepted": float(result.accepted),
                        "speculator/train_loss": float(
                            result.train_loss if result.train_loss is not None else float("nan")
                        ),
                        "speculator/candidate_holdout_loss": float(
                            result.candidate_holdout_loss if result.candidate_holdout_loss is not None else float("nan")
                        ),
                        "speculator/candidate_holdout_agreement": float(
                            result.candidate_holdout_agreement
                            if result.candidate_holdout_agreement is not None
                            else float("nan")
                        ),
                        "speculator/train_duration_seconds": float(
                            result.duration_seconds if result.duration_seconds is not None else float("nan")
                        ),
                    }
                )
                if result.error is not None:
                    self._speculator_update_failures += 1
                    logger.warning(
                        "DraftTrainer update failed: step={} error={}",
                        result.step,
                        result.error,
                    )
                else:
                    logger.info(
                        "DraftTrainer update completed: step={} accepted={} revision={} uri={}",
                        result.step,
                        result.accepted,
                        result.draft_revision,
                        result.candidate_uri,
                    )
            else:
                self.all_metrics["speculator/update_pending"] = 1.0
        self.all_metrics["speculator/update_failures"] = float(self._speculator_update_failures)
        await self._refresh_latest_speculator()

    async def _sync_policy_for_rollouts(self, *, reason: str) -> None:
        """Load the policy's current weights into the inference engines.

        Colocated engines sleep while the policy trains; they wake for the weights and then take the GPU
        memory the policy model releases. Separate engines keep serving rollouts, so they pause for the sync
        and their requests in flight resume on the new weights.
        """
        timings = self.all_startup_timings if reason == "initial" else self.all_timings
        with Timer("sync_weights", timings) as update_timer:
            if self.colocate_all:
                await asyncio.to_thread(self.policy_model.offload_to_cpu, offload_optimizer=True, offload_model=False)
                await self.inference_engine_client.wake_up(tags=["weights"])
                await self.sync_policy_weights_to_inference_engines()
                await asyncio.to_thread(self.policy_model.offload_to_cpu, offload_optimizer=False, offload_model=True)
                await self.inference_engine_client.wake_up(tags=["kv_cache"])
            else:
                # Expert-block sync writes into live engine parameters, so the initial sync pauses generation too.
                pause = reason != "initial" or self._expert_block_sync is not None
                if pause:
                    await self.inference_engine_client.pause_generation()
                # Training backloads the optimizer before every step. Offload it after every update, including
                # the initial sync, so Megatron gradient buffers are not resized while still on the GPU.
                await asyncio.to_thread(
                    self._offload_policy_optimizer, timings, timer_label="offload_policy_optimizer_to_cpu"
                )
                await self.sync_policy_weights_to_inference_engines()
                if pause:
                    await self.inference_engine_client.resume_generation()
        self._log_weight_update_completed(reason=reason, duration_seconds=update_timer.duration)

    async def sync_policy_weights_to_inference_engines(self) -> None:
        # Align policy actors before the weight extraction collectives.
        await self._drain_policy_event_loops()
        if self._expert_block_sync is not None:
            timings = await self._expert_block_sync.sync(self.global_step)
            self.all_timings.update(timings.as_metrics())
            logger.info(
                "Expert-block sync: step={} install_seconds={:.3f} policy_seconds={:.3f} receiver_seconds={:.3f} "
                "expert_seconds={:.3f} dense_seconds={:.3f}",
                self.global_step,
                timings.install_seconds,
                timings.policy_seconds,
                timings.receiver_seconds,
                timings.expert_seconds,
                timings.dense_seconds,
            )
            if self.cfg.generator.expert_block_sync.verify:
                self.all_timings.update(await self._expert_block_sync.verify(self.global_step))
        else:
            await self.policy_model.async_run_method(
                "pass_through", "broadcast_to_inference_engines", self.inference_engine_client
            )
        # A hard sync point leaves every policy rank free before the next forward.
        await self._drain_policy_event_loops()

    async def _drain_policy_event_loops(self):
        """Wait for every actor loop to finish pending weight-sync work.

        The next synchronous forward requires all model ranks to enter its
        collectives. An async barrier lets each actor finish its prior coroutine
        before the driver sends that forward.
        """
        refs = self.policy_model.async_run_ray_method("pass_through", "barrier_all")
        await asyncio.gather(*refs)

    def _offload_policy_optimizer(self, timings: dict, *, timer_label: str) -> None:
        """Move optimizer state off GPU before rollout generation or checkpoint restore."""
        if not self.colocate_all and self.cfg.trainer.offload_optimizer_during_rollouts:
            with Timer(timer_label, timings):
                self.policy_model.offload_to_cpu(offload_optimizer=True, offload_model=False)

    def _log_weight_update_completed(self, *, reason: str, duration_seconds: float) -> None:
        if self._training_metrics_enabled:
            record_event(
                "weight_sync_completed",
                {"model_version_step": self.global_step, "duration_seconds": duration_seconds},
                attributes={"role": TRAINER_ROLE, "step": str(self.global_step), "reason": reason},
            )
        logger.info(
            "Policy weights updated: step={} reason={} duration_seconds={:.3f}",
            getattr(self, "global_step", None),
            reason,
            duration_seconds,
        )

    def _log_optimizer_step_completed(
        self,
        *,
        epoch: int,
        training_input: TrainingInputBatch,
        duration_seconds: float,
    ) -> None:
        self.all_metrics.update(training_input.metadata["consumed_stop_metrics"])
        if self._training_metrics_enabled:
            record_consumed_work(consumed_work(training_input), step=self.global_step)
        logger.info(
            "Optimizer step completed: step={} epoch={} sequences={} duration_seconds={:.3f}",
            self.global_step,
            epoch,
            len(training_input["sequences"]),
            duration_seconds,
        )

    def _log_training_step_completed(self, *, epoch: int, duration_seconds: float) -> None:
        logger.info(
            "Training step completed: step={} total_steps={} epoch={} duration_seconds={:.3f}",
            self.global_step,
            self.total_training_steps,
            epoch,
            duration_seconds,
        )

    async def _train_loop(self):
        await self._init_weight_sync()

        # Colocated training loads the policy model onto the GPU before its checkpoint.
        if self.colocate_all:
            self.policy_model.backload_to_gpu()

        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints", self.all_startup_timings):
                self.global_step, _ = self.load_checkpoints()
            logger.info(f"Resumed training from global_step {self.global_step}")
            if self._restored_rollout_state is not None:
                await self.context.load_state_dict(self._restored_rollout_state)
                self._restored_rollout_state = None

        await self._start_draft_trainer()
        await self._sync_policy_for_rollouts(reason="initial")

        # Synchronize before checking completion so a requested final evaluation uses the checkpoint weights.
        # The loaded global_step counts completed steps, so >= treats a resume exactly at max_steps as complete.
        if self.resume_mode != ResumeMode.NONE and self.global_step >= self.total_training_steps:
            await self._handle_resume_at_max_steps()
            return

        self._log_startup_timings()
        self._record_run_configuration()

        if self.cfg.trainer.algorithm.use_kl_in_reward:
            self.reward_kl_controller = get_kl_controller(self.cfg.trainer.algorithm)

        self._control.reset()
        self._control = await self.callback_handler.call_event_async(
            "on_train_begin",
            self._create_trainer_state(epoch=self.global_step // self.num_steps_per_epoch),
            self._control,
            trainer=self,
        )
        if self._control.should_evaluate and self.eval_dataset is not None:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self._log_metrics_stdout(eval_metrics, step=self.global_step, kind="eval")
                self.tracker.log(eval_metrics, step=self.global_step, commit=True)
            self._control.should_evaluate = False

        self.context.start()
        await self._publish(self.global_step + 1)

        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Step Progress")
        last_completed_step = self.global_step
        while self.global_step < self.total_training_steps:
            self.global_step += 1
            epoch = (self.global_step - 1) // self.num_steps_per_epoch
            cycle_started = time.perf_counter()
            with Timer("step", self.all_timings) as step_timer:
                training_input = await self._train_step(epoch)
                # The core wall excludes the step-end callbacks below.
                core_seconds = time.perf_counter() - cycle_started
                step_state = self._create_trainer_state(epoch, active_step_duration=step_timer.elapsed)
                await self._run_step_end_callbacks(step_state)
            self._update_step_performance_metrics(
                training_input, core_seconds=core_seconds, cycle_started=cycle_started
            )

            if self._control.should_log:
                log_payload = {
                    **self.all_metrics,
                    **{f"timing/{k}": v for k, v in self.all_timings.items()},
                    **get_system_memory_metrics(),
                }
                self._log_metrics_stdout(log_payload, step=self.global_step, kind="train")
                self.tracker.log(log_payload, step=self.global_step, commit=self.cfg.trainer.tracker_commit_each_step)
                await self.callback_handler.call_event_async(
                    "on_log", step_state, self._control, logs=log_payload, trainer=self
                )
            self._log_training_step_completed(epoch=epoch, duration_seconds=step_timer.duration)

            self._step_time_history.append(step_timer.duration)
            self.all_metrics = {}
            publish_step_timings(self.all_timings, self.global_step)
            self.all_timings = {}
            pbar.update(1)
            last_completed_step = self.global_step
            record_policy_step(self.global_step)

            stop = self._control.should_training_stop
            if step_state.is_epoch_end:
                await self._end_epoch(epoch)
                stop = stop or self._control.should_training_stop
            if stop:
                logger.info("Training stopped early by callback")
                break
            if self.global_step < self.total_training_steps:
                await self._publish(self.global_step + 1)
        pbar.close()

        await self._finalize_training(completed_step=last_completed_step, epoch=self.cfg.trainer.epochs - 1)
        logger.info("Training done!")

    async def _init_weight_sync(self) -> None:
        with Timer("init_weight_sync_state", self.all_startup_timings):
            if self.cfg.generator.weight_sync_transport != WeightSyncTransport.EXPERT_BLOCK:
                self.init_weight_sync_state()
                return
            self._expert_block_sync = ExpertBlockSync(
                policy_model=self.policy_model,
                inference_engine_client=self.inference_engine_client,
                timeout_seconds=self.cfg.generator.expert_block_sync.timeout_seconds,
            )
            for phase, seconds in (await self._expert_block_sync.prepare()).items():
                self.all_startup_timings[f"expert_block_sync/{phase}"] = seconds

    def _record_run_configuration(self) -> None:
        if not self._training_metrics_enabled:
            return
        placement = self.cfg.trainer.placement
        megatron = self.cfg.trainer.policy.get("megatron_config", {})
        buffer = self.context.config
        # The field names are the ones the async dashboard reads.
        record_event(
            "async_run_configuration",
            {
                "strategy": self.cfg.trainer.strategy,
                "policy_nodes": placement.policy_num_nodes,
                "policy_gpus_per_node": placement.policy_num_gpus_per_node,
                "policy_tp": megatron.get("tensor_model_parallel_size"),
                "policy_pp": megatron.get("pipeline_model_parallel_size"),
                "policy_cp": megatron.get("context_parallel_size"),
                "policy_ep": megatron.get("expert_model_parallel_size"),
                "generation_workers": buffer.max_in_flight,
                "mini_batch_size": buffer.batch_size,
                "max_staleness_steps": buffer.max_staleness_steps,
            },
            attributes={"role": TRAINER_ROLE, "step": str(self.global_step)},
        )

    def _update_step_performance_metrics(
        self, training_input: TrainingInputBatch, *, core_seconds: float, cycle_started: float
    ) -> None:
        if not self._training_metrics_enabled:
            return
        placement = self.cfg.trainer.placement
        generator = self.cfg.generator
        consumed = consumed_work(training_input)
        self.all_metrics.update(
            async_step_metrics(
                core_seconds=core_seconds,
                cycle_seconds=time.perf_counter() - cycle_started,
                buffer_wait_seconds=self.all_timings["wait_for_generation_buffer"],
                training_seconds=self.all_timings["run_training"],
                sync_seconds=self.all_timings["sync_weights"],
                consumed_loss_tokens=consumed.loss_tokens,
                consumed_response_tokens=consumed.response_tokens,
                policy_gpus=placement.policy_num_nodes * placement.policy_num_gpus_per_node,
                inference_gpus=(
                    generator.num_inference_engines
                    * generator.inference_engine_tensor_parallel_size
                    * generator.inference_engine_pipeline_parallel_size
                    * generator.inference_engine_data_parallel_size
                ),
            )
        )

    async def _publish(self, step: int) -> None:
        """Open rollout leases for training step ``step``, whose policy weights the engines now hold."""
        await self._begin_speculator_capture(step)
        await self.context.publish(step)

    async def _train_step(self, epoch: int) -> TrainingInputBatch:
        """Read this step's batch from the rollout buffer, train on it, sync the new weights, and return the batch."""
        logger.info(
            "Rollout batch started: step={} required_groups={}", self.global_step, self.context.config.batch_size
        )
        with (
            Timer("wait_for_generation_buffer", self.all_timings) as rollout_wait_timer,
            critical_phase("rollout_or_inference_wait", self.global_step),
            async_phase_window("rollout_wait", step=self.global_step, enabled=self._rollout_spans_enabled),
        ):
            groups, selection_metrics = await self.context.next_batch(
                stall_timeout=self._rollout_stall_timeout(),
                on_admitted=self._submit_admitted_groups_for_teacher_scoring,
            )
        self.all_metrics.update(selection_metrics)
        await self._seal_speculator_capture()
        await self._start_speculator_update()
        if self.colocate_all:
            # Colocation runs at staleness 0, where no rollout is in flight once the batch is complete.
            await self.inference_engine_client.sleep()

        scored_distillation = None
        if self._distillation_runtime is not None:
            with Timer("wait_for_teacher_evidence", self.all_timings):
                scored_distillation = await self._await_admitted_teacher_evidence(groups)

        training_input = await asyncio.to_thread(self.convert_rollout_groups_to_training_input, groups)
        if scored_distillation is not None:
            self.all_metrics.update(
                self._distillation_runtime.attach_to_training_input(training_input, scored_distillation)
            )
            self.all_metrics.update(
                {
                    "distillation/teacher_count": float(
                        len({route.teacher_id for scored in scored_distillation for route in scored.routes})
                    ),
                    DISTILLATION_SCORED_TOKENS_METRIC: float(
                        sum(scored.distillation.valid_mask.sum().item() for scored in scored_distillation)
                    ),
                }
            )
        response_ids = [response for group in groups for response in group.trajectory_batch["response_ids"]]
        logger.info(
            "Rollout batch completed: step={} groups={} trajectories={} response_tokens={} staleness_mean={:.3f} "
            "staleness_max={} duration_seconds={:.3f}",
            self.global_step,
            len(groups),
            len(response_ids),
            sum(len(response) for response in response_ids),
            self.all_metrics["async/staleness_mean"],
            self.all_metrics["async/staleness_max"],
            rollout_wait_timer.duration,
        )

        # Records whether this batch lacked rollout logprobs, so TIS fell back to the standard policy loss.
        # A skipped fraction near 1.0 means rollout-logprob capture is broken.
        if self.cfg.trainer.algorithm.use_tis:
            batch_skipped = float(getattr(self, "_tis_batch_skipped_no_logprobs", 0.0))
            self._tis_skipped_count = getattr(self, "_tis_skipped_count", 0.0) + batch_skipped
            self._tis_total_count = getattr(self, "_tis_total_count", 0.0) + 1.0
            self.all_metrics.update(
                {
                    "tis/batch_skipped_no_logprobs": batch_skipped,
                    "tis/skipped_fraction": self._tis_skipped_count / self._tis_total_count,
                }
            )

        with (
            Timer("run_training", self.all_timings),
            async_phase_window("training", step=self.global_step, enabled=self._rollout_spans_enabled),
        ):
            status = await self._run_training(training_input)
        self._log_optimizer_step_completed(
            epoch=epoch,
            training_input=training_input,
            duration_seconds=self.all_timings["train_critic_and_policy"],
        )

        await self._poll_speculator_lifecycle()
        with async_phase_window("weight_sync", step=self.global_step, enabled=self._rollout_spans_enabled):
            await self._sync_policy_for_rollouts(reason="training_step")

        logger.info(status)
        self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
        return training_input

    async def _end_epoch(self, epoch: int) -> None:
        self._control.reset()
        self._control = await self.callback_handler.call_event_async(
            "on_epoch_end", self._create_trainer_state(epoch=epoch), self._control, trainer=self
        )
        if self.global_step == self.total_training_steps or self.ref_model is None:
            return
        ref_callback = self._get_ref_update_callback()
        if ref_callback is not None and ref_callback.should_update_ref:
            with Timer("update_ref_with_policy", self.all_timings):
                await asyncio.to_thread(self.update_ref_with_policy)

    async def _run_training(self, training_input: TrainingInputBatch):
        # The drain after the last weight sync can go stale while the rollout buffer fills, so align the
        # policy actor loops immediately before every forward.
        await self._drain_policy_event_loops()
        with Timer("fwd_logprobs_values_reward", self.all_timings):
            training_input = await asyncio.to_thread(self.fwd_logprobs_values_reward, training_input)

        if self.cfg.trainer.algorithm.use_kl_in_reward:
            with Timer("apply_reward_kl_penalty", self.all_timings):
                training_input = self.apply_reward_kl_penalty(training_input)

        with Timer("compute_advantages_and_returns", self.all_timings):
            training_input = self.compute_advantages_and_returns(training_input)
            training_input = self.finalize_advantages_for_training(training_input)

        if self.cfg.trainer.dump_data_batch:
            with Timer("dump_data_batch", self.all_timings):
                self.dump_data(training_input, file_name=f"global_step_{self.global_step}_training_input")

        with Timer("train_critic_and_policy", self.all_timings), critical_phase("train_step", self.global_step):
            status = await asyncio.to_thread(self.train_critic_and_policy, training_input)

        return status

    def _rollout_stall_timeout(self) -> float:
        """Adaptive deadline between admitted groups.

        An explicit timeout is returned unchanged. Otherwise the deadline is a
        multiple of the recent median step time (at least 10 minutes); before
        step timing exists, it is 30 minutes.
        """
        return AdmissionProgressWatchdog.start(
            now=0.0,
            recent_step_times=self._step_time_history,
            timeout_override=self.group_admission_stall_timeout,
        ).timeout

    def _group_for_teacher_scoring(self, group: RolloutGroup) -> TrajectoryBatch:
        """Apply the learner's row selector without duplicating its metric side effects."""
        if self.trajectory_selector is None:
            return group.trajectory_batch
        row_count = len(group.trajectory_batch["response_ids"])
        return self.trajectory_selector.select(group.trajectory_batch, [group.uid] * row_count).trajectory_batch

    async def _submit_admitted_groups_for_teacher_scoring(self, groups: List[RolloutGroup]) -> None:
        """Enqueue newly admitted groups, letting bounded teacher queues backpressure admission."""
        runtime = self._distillation_runtime
        if runtime is None:
            return
        for group in groups:
            self._distillation_tickets[group.uid] = await runtime.submit_before_batch_assembly(
                self._group_for_teacher_scoring(group)
            )

    async def _await_admitted_teacher_evidence(
        self,
        groups: List[RolloutGroup],
    ) -> tuple[RoutedScoredDistillationBatch, ...]:
        """Gate learner batch assembly on every admitted group's teacher evidence."""
        missing = [group.uid for group in groups if group.uid not in self._distillation_tickets]
        if missing:
            raise RuntimeError(f"admitted groups are missing teacher score tickets: {missing}")
        scored = tuple(await asyncio.gather(*(self._distillation_tickets[group.uid].result() for group in groups)))
        for group, scored_group in zip(groups, scored, strict=True):
            selected_group = self._group_for_teacher_scoring(group)
            trajectory_ids = selected_group.get("trajectory_ids")
            if trajectory_ids is None:
                raise ValueError("teacher scoring requires stable trajectory IDs on admitted groups")
            expected_ids = tuple(trajectory_id.to_string() for trajectory_id in trajectory_ids)
            if scored_group.trajectory_ids != expected_ids:
                raise ValueError(
                    f"teacher evidence row order does not match admitted group {group.uid!r}: "
                    f"expected={expected_ids}, scored={scored_group.trajectory_ids}"
                )
            del self._distillation_tickets[group.uid]
        return scored

    def convert_rollout_groups_to_training_input(self, groups: List[RolloutGroup]) -> TrainingInputBatch:
        """Concatenate one batch of admitted groups and convert it to a training batch."""
        batch_size = self.context.config.batch_size
        max_staleness_steps = self.context.config.max_staleness_steps
        assert len(groups) == batch_size, f"Expected {batch_size} groups, got {len(groups)}"
        with Timer("assemble_generation_group_mini_batch", self.all_timings):
            uids = [group.uid for group in groups for _ in group.trajectory_batch["response_ids"]]
            stalenesses = [self.global_step - group.policy_step for group in groups]
            staleness_by_uid = {group.uid: staleness for group, staleness in zip(groups, stalenesses, strict=True)}
            record_rollout_staleness(stalenesses, self.global_step)
            assert max(stalenesses) <= max_staleness_steps, (
                f"batch assembly returned staleness {max(stalenesses)} above max {max_staleness_steps}"
            )

            trajectory_batch = concatenate_trajectory_batches(
                [group.trajectory_batch for group in groups],
                require_rollout_logprobs=policy_loss_requires_rollout_logprobs(
                    self.cfg.trainer.algorithm.policy_loss_type
                ),
                tis_lcs_alert_threshold=float(self.cfg.trainer.algorithm.tis_lcs_alert_threshold),
            )
            assert trajectory_batch["rollout_metrics"] is not None, "Rollout metrics should be non-null."
            self.all_metrics.update(trajectory_batch["rollout_metrics"])
            self.all_metrics.update(
                {
                    "async/staleness_mean": sum(stalenesses) / len(stalenesses),
                    "async/staleness_max": max(stalenesses),
                    "async/staleness_min": min(stalenesses),
                    "async/staleness_ratio": sum(1 for s in stalenesses if s > 0) / len(stalenesses),
                }
            )

        with Timer("postprocess_trajectory_batch", self.all_timings):
            trajectory_batch = self.postprocess_trajectory_batch(trajectory_batch, uids)
            trajectory_batch, uids = self.select_trajectories(trajectory_batch, uids)
        # Built after selection, because select_trajectories may drop rows.
        rollout_staleness = [staleness_by_uid[uid] for uid in uids]

        logger.debug(f"Example generated: {self.tokenizer.decode(trajectory_batch['response_ids'][0])}")

        with Timer("convert_to_training_input", self.all_timings):
            training_input = self.convert_to_training_input(trajectory_batch, uids, rollout_staleness=rollout_staleness)
        if self._training_metrics_enabled:
            self._record_consumed_staleness(uids, rollout_staleness, training_input["response_mask"][: len(uids)])
        return training_input

    def _record_consumed_staleness(
        self, uids: List[str], rollout_staleness: List[int], response_masks: torch.Tensor
    ) -> None:
        counts: dict[str, dict[str, int]] = {}
        for uid, steps, mask in zip(uids, rollout_staleness, response_masks, strict=True):
            group = counts.setdefault(uid, {"staleness": steps, "groups": 1, "sequences": 0, "response_tokens": 0})
            group["sequences"] += 1
            group["response_tokens"] += int(mask.sum().item())
        for group in counts.values():
            record_event("consumed_staleness", group, attributes={"role": TRAINER_ROLE, "step": str(self.global_step)})

    def build_models(self, PolicyWorker, CriticWorker, RefWorker, policy_pg: Optional[PlacementGroup] = None):
        """
        Initialize the actors for training, and handle colocation logic

        Args:
            policy_pg: Optional pre-reserved placement group dedicated to the
                policy/training actors. Supplied (non-None) only for the
                disaggregated, no-ref case when `placement.policy_strict_spread_pg`
                is enabled — it is a STRICT_SPREAD whole-node placement group
                reserved BEFORE the inference engines so that policy and
                inference land on disjoint nodes. When None, the legacy
                lazy-PACK behavior in `PPORayActorGroup._initiate_actors` is used.
        """
        cfg = self.cfg
        pg = None

        use_ref_model = cfg.trainer.algorithm.use_kl_loss or cfg.trainer.algorithm.use_kl_in_reward

        if cfg.trainer.placement.colocate_all:
            num_policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
            num_critic_gpus = cfg.trainer.placement.critic_num_gpus_per_node * cfg.trainer.placement.critic_num_nodes
            num_ref_gpus = cfg.trainer.placement.ref_num_gpus_per_node * cfg.trainer.placement.ref_num_nodes
            num_rollout_gpus = (
                cfg.generator.num_inference_engines
                * cfg.generator.inference_engine_tensor_parallel_size
                * cfg.generator.inference_engine_pipeline_parallel_size
                * cfg.generator.inference_engine_data_parallel_size
            )
            assert num_policy_gpus == num_rollout_gpus, (
                "num_policy_gpus and num_rollout_gpus must be the same when colocating all models"
            )
            pg = self.colocate_pg

            policy_model = PPORayActorGroup(
                cfg,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
                colocate_all=True,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            if use_ref_model:
                assert num_policy_gpus == num_ref_gpus, (
                    "num_policy_gpus and num_ref_gpus must be the same when colocating policy and ref model"
                )
                ref_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2 if pg else 1,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
            else:
                ref_model = None

            if cfg.trainer.critic.model.path:
                assert num_policy_gpus == num_critic_gpus, (
                    "num_policy_gpus and num_critic_gpus must be the same when colocating policy and critic model"
                )
                critic_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

        else:
            if cfg.trainer.placement.colocate_policy_ref and use_ref_model:
                assert (
                    cfg.trainer.placement.policy_num_nodes == cfg.trainer.placement.ref_num_nodes
                    and cfg.trainer.placement.policy_num_gpus_per_node == cfg.trainer.placement.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate policy and ref model."

                bundles = [
                    {
                        "GPU": cfg.trainer.placement.policy_num_gpus_per_node,
                        "CPU": cfg.trainer.placement.policy_num_gpus_per_node,
                    }
                    for _ in range(cfg.trainer.placement.policy_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                get_ray_pg_ready_with_timeout(
                    pg, timeout=int(self.cfg.trainer.distributed.placement_group_timeout_seconds)
                )

            # Dedicated, pre-reserved STRICT_SPREAD policy placement group
            # (disaggregated no-ref case only). Supplied (non-None) by the
            # entrypoint when `placement.policy_strict_spread_pg` is enabled.
            # It is guaranteed not to coincide with the colocate_policy_ref
            # branch above because eligibility requires use_ref_model=False.
            # Each policy actor takes a full GPU within its node's whole-node
            # bundle (so the inference engines, on disjoint nodes, never share
            # a physical GPU with a policy worker).
            if policy_pg is not None:
                assert not use_ref_model, "dedicated policy_pg is only used when no ref model is present"
                assert pg is None, "dedicated policy_pg must not coexist with a shared policy/ref pg"
                pg = policy_pg

            # Pin each policy actor to its Ray-assigned physical GPU only when
            # the dedicated policy PG uses per-GPU {GPU:1} bundles (each actor
            # then owns one bundle == one GPU, so ray.get_gpu_ids()[0] is a
            # distinct, reliable physical id even when the SIF Ray leaves
            # CUDA_VISIBLE_DEVICES unmasked). Off otherwise → unchanged pinning.
            _policy_pin_to_ray_gpu_id = policy_pg is not None and policy_per_gpu_bundles_enabled(cfg)
            # Deterministic forced-CVD-mask pin (opt-in; default false). Only
            # meaningful alongside the per-GPU-bundle pin. Masks each policy
            # actor to its single physical GPU before any CUDA/EP-mesh init.
            _policy_force_cvd_mask = _policy_pin_to_ray_gpu_id and policy_force_cvd_mask_enabled(cfg)
            policy_model = PPORayActorGroup(
                cfg,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=(1 if policy_pg is not None else (0.75 if pg else 1)),
                colocate_all=False,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                pin_to_ray_gpu_id=_policy_pin_to_ray_gpu_id,
                force_cvd_mask=_policy_force_cvd_mask,
            )
            if use_ref_model:
                ref_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.25 if pg else 1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
            else:
                ref_model = None

            if cfg.trainer.critic.model.path:
                critic_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    num_gpus_per_actor=1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

        self.policy_model: PPORayActorGroup = policy_model
        self.critic_model: Optional[PPORayActorGroup] = critic_model
        self.ref_model: Optional[PPORayActorGroup] = ref_model
        self._initialize_model_actors(cfg, policy_model, critic_model, ref_model)

        logger.info("init policy/ref/critic models done")

    def _initialize_model_actors(
        self,
        cfg: DictConfig,
        policy_model: PPORayActorGroup,
        critic_model: Optional[PPORayActorGroup],
        ref_model: Optional[PPORayActorGroup],
    ) -> None:
        """Initialize all model actors within one shared wall-clock deadline."""
        initialization_deadline = time.monotonic() + _MODEL_INITIALIZATION_TIMEOUT

        if not cfg.trainer.placement.colocate_all:
            refs = []
            if ref_model is not None:
                refs.extend(ref_model.async_init_model(cfg.trainer.ref.model.path))
            refs.extend(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=self.total_training_steps,
                )
            )
            if cfg.trainer.critic.model.path:
                assert critic_model is not None
                refs.extend(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=self.total_training_steps,
                    )
                )
            self._wait_for_setup_phase(
                refs,
                deadline=initialization_deadline,
                phase="policy/ref/critic model initialization",
            )
            self._wait_for_setup_phase(
                policy_model.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id),
                deadline=initialization_deadline,
                phase="policy model finalization",
            )
        else:
            if ref_model is not None:
                self._wait_for_setup_phase(
                    ref_model.async_init_model(cfg.trainer.ref.model.path),
                    deadline=initialization_deadline,
                    phase="reference model initialization",
                )
                self._wait_for_setup_phase(
                    ref_model.offload_to_cpu(nonblocking=True),
                    deadline=initialization_deadline,
                    phase="reference model offload",
                )
            self._wait_for_setup_phase(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=self.total_training_steps,
                ),
                deadline=initialization_deadline,
                phase="policy model initialization",
            )
            self._wait_for_setup_phase(
                policy_model.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id),
                deadline=initialization_deadline,
                phase="policy model finalization",
            )
            self._wait_for_setup_phase(
                policy_model.offload_to_cpu(nonblocking=True),
                deadline=initialization_deadline,
                phase="policy model offload",
            )
            if cfg.trainer.critic.model.path:
                assert critic_model is not None
                self._wait_for_setup_phase(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=self.total_training_steps,
                    ),
                    deadline=initialization_deadline,
                    phase="critic model initialization",
                )
                self._wait_for_setup_phase(
                    critic_model.offload_to_cpu(nonblocking=True),
                    deadline=initialization_deadline,
                    phase="critic model offload",
                )

    def _wait_for_setup_phase(self, refs, *, deadline: float, phase: str):
        """Wait for one setup phase and terminate all actors if the shared deadline expires."""
        remaining_seconds = max(0.0, deadline - time.monotonic())
        try:
            return ray.get(refs, timeout=remaining_seconds)
        except ray.exceptions.GetTimeoutError as error:
            message = (
                f"{phase} timed out after {_MODEL_INITIALIZATION_TIMEOUT} seconds; "
                "terminating model actors and failing the training job"
            )
            logger.error(message)
            self._kill_ray_actors()
            raise RuntimeError(message) from error

    def init_weight_sync_state(self):
        """
        Setup the connection between policy model and inference engine for weight syncing.
        """
        # Diagnostic: unwrap un-pickleable Ray exceptions into a plain
        # RuntimeError so a recurrence reports the TRUE cause (e.g. a raylet
        # killed by a GPFS SIGBUS/ESTALE mmap fault -> ActorUnavailableError)
        # rather than the secondary PicklingError / pydantic_compat
        # ModuleNotFoundError that arises from Ray failing to re-serialize the
        # dynamically-generated RayTaskError across the dying boundary. Happy
        # path unchanged.
        try:
            ray.get(
                self.policy_model.async_run_ray_method(
                    "pass_through", "init_weight_sync_state", self.inference_engine_client
                )
            )
        except ray.exceptions.RayError as e:
            raise RuntimeError(f"init_weight_sync_state failed at Ray boundary: {e!r}") from None
        logger.info(
            "Weight sync ready: policy_workers={} inference_engines={}",
            len(self.policy_model.actor_infos),
            len(self.inference_engine_client.engines),
        )

    def _resolve_num_experts(self) -> Optional[int]:
        """Resolve the policy model's MoE expert count from its HF config, memoized.

        Used to pick a DETERMINISTIC (rank-invariant) dtype for the
        rollout_routed_experts transport tensor in the collator — see
        ``convert_prompts_responses_to_batch_tensors``. Reads the HF config at
        ``cfg.trainer.policy.model.path`` once and resolves the expert count across
        MoE arch variants (Qwen3-MoE: ``num_experts``; Mixtral: ``num_local_experts``;
        DeepSeek/Qwen-style: ``n_routed_experts``). Returns None (and caches None) if
        the config has no such field (non-MoE) or can't be loaded, in which case the
        collator falls back to its non-deterministic per-batch-max pick with a warning.
        """
        if getattr(self, "_num_experts_cache", "__unset__") != "__unset__":
            return self._num_experts_cache
        num_experts: Optional[int] = None
        try:
            from transformers import AutoConfig

            model_path = self.cfg.trainer.policy.model.path
            hf_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                revision=self.cfg.trainer.policy.model.get("revision"),
            )
            # Some families nest the expert count under a text/decoder sub-config.
            candidates = [hf_config, getattr(hf_config, "text_config", None)]
            for cfg_obj in candidates:
                if cfg_obj is None:
                    continue
                for attr in ("num_experts", "num_local_experts", "n_routed_experts"):
                    val = getattr(cfg_obj, attr, None)
                    if isinstance(val, int) and val > 0:
                        num_experts = val
                        break
                if num_experts is not None:
                    break
        except Exception as e:  # noqa: BLE001 — config load is best-effort; fallback is safe
            logger.warning(
                f"Could not resolve num_experts from policy model config for the "
                f"routed-experts dtype pick ({e!r}); the collator will use its "
                f"non-deterministic per-batch-max fallback."
            )
            num_experts = None
        if num_experts is not None:
            logger.info(f"Resolved policy model num_experts={num_experts} for routed-experts dtype pick.")
        self._num_experts_cache: Optional[int] = num_experts
        return num_experts

    def convert_to_training_input(
        self,
        trajectory_batch: TrajectoryBatch,
        uids: List[str],
        *,
        rollout_staleness: List[int] | None = None,
    ) -> TrainingInputBatch:
        """Converts lists to a padded batch of tensors for training"""
        assert_training_groups_eligible(trajectory_batch, uids, self.group_advantage_invariant)
        prompt_ids: List[List[int]] = trajectory_batch["prompt_token_ids"]
        response_ids: List[List[int]] = trajectory_batch["response_ids"]
        rewards: List[List[float]] = trajectory_batch["rewards"]
        loss_masks: List[List[int]] = trajectory_batch["loss_masks"]

        logprobs: Optional[List[List[float]]] = trajectory_batch.get("rollout_logprobs", None)

        # MoE router-replay capture rail (Stage 1): only pull routed_experts when
        # the flag is on. Gated so the flag-off TrainingInputBatch is byte-identical
        # (the field is never even passed to the collator nor set on the batch).
        moe_router_replay = moe_router_replay_requested(self.cfg)
        routed_experts = trajectory_batch.get("rollout_routed_experts", None) if moe_router_replay else None
        # Deterministic dtype for the rollout_routed_experts transport tensor:
        # resolve the model's expert count once (memoized) and pass it to the
        # collator so the narrowed dtype is keyed on num_experts (max possible id),
        # NOT the per-batch observed max — otherwise ranks whose batch max straddles
        # a dtype boundary diverge and a later collective on this tensor hangs NCCL.
        # Only needed when we actually carry routed experts (moe_router_replay on).
        num_experts = self._resolve_num_experts() if routed_experts is not None else None

        # Loop-behavior reward shaping (Stage B / F5 + F4): only pull the per-token
        # shaping channel + span tags when the channel is enabled. Gated so the
        # flag-off TrainingInputBatch is byte-identical (the fields are never passed
        # to the collator nor set on the batch). Mirrors moe_router_replay above.
        enable_token_reward_channel = bool(self.cfg.trainer.algorithm.get("enable_token_reward_channel", False))
        token_level_shaping = trajectory_batch.get("token_level_shaping", None) if enable_token_reward_channel else None
        response_span_tags = trajectory_batch.get("response_span_tags", None) if enable_token_reward_channel else None
        loop_advantages = trajectory_batch.get("loop_advantages")

        (
            sequences_tensor,
            attention_masks_tensor,
            response_masks_tensor,
            rewards_tensor,
            loss_masks_tensor,
            rollout_logprobs_tensor,
            rollout_routed_experts_tensor,
            token_level_shaping_tensor,
            response_span_tags_tensor,
        ) = convert_prompts_responses_to_batch_tensors(
            self.tokenizer,
            prompt_ids,
            response_ids,
            rewards,
            loss_masks,
            logprobs,
            routed_experts,
            token_level_shaping,
            response_span_tags,
            num_experts,
        )
        behavior_logprobs_required = policy_loss_requires_rollout_logprobs(self.cfg.trainer.algorithm.policy_loss_type)
        if behavior_logprobs_required and rollout_logprobs_tensor is None:
            raise ValueError("rollout_logprobs are required for behavior_clip policy loss")

        # sanity check for tis
        #
        # Graceful TIS degrade (Fix A, 2026-06-07): when use_tis is on but the
        # ENTIRE training batch came back with no rollout logprobs
        # (rollout_logprobs_tensor is None), do NOT hard-assert/crash. The
        # runner already detects + logs the all-None case ("ALL N
        # trajectories missing logprobs. This batch cannot be used for TIS
        # training"); the trainer must complete the hardening by degrading to
        # standard (non-TIS) policy loss for THIS batch only. The None tensor
        # propagates cleanly: TensorBatch.chunk/slice leave None values
        # un-chunked (-> each micro-batch sees rollout_logprobs=None), the
        # worker's TIS diagnostics already guard `is not None`, and
        # ppo_policy_loss skips the TIS importance ratio when rollout_logprobs
        # is None (see policy_losses.ppo_policy_loss). We surface the skip via a
        # `tis/batch_skipped_no_logprobs` metric (set on the driver below) so the
        # failure mode is observable; the relaunch skip-fraction is the live
        # systematic-vs-intermittent diagnostic.
        self._tis_batch_skipped_no_logprobs = 0.0
        if self.cfg.trainer.algorithm.use_tis:
            if rollout_logprobs_tensor is None:
                self._tis_batch_skipped_no_logprobs = 1.0
                logger.warning(
                    "use_tis is True but this training batch has NO rollout logprobs "
                    "(all-None). Degrading to standard (non-TIS) policy loss for THIS "
                    "batch and continuing. tis/batch_skipped_no_logprobs=1. If this "
                    "persists (skip-fraction ~1.0) the rollout-logprob capture is "
                    "systematically broken (e.g. routed_experts capture displacing "
                    "logprobs); a low/intermittent rate is context-length errors."
                )
            else:
                assert rollout_logprobs_tensor.shape == loss_masks_tensor.shape, "Logprobs should look like responses"
        # Stage 1 invariant (scope Q3 #2): routed_experts.shape[:2] == loss_mask.shape.
        if rollout_routed_experts_tensor is not None:
            assert rollout_routed_experts_tensor.shape[:2] == loss_masks_tensor.shape, (
                "routed_experts response axis should look like responses"
            )
        distillation_tensors = _validated_distillation_tensors(trajectory_batch, response_masks_tensor)
        training_input = TrainingInputBatch(
            {
                "sequences": sequences_tensor,  # Full trajectories (padded and concatenated prompts and responses)
                "attention_mask": attention_masks_tensor,
                "response_mask": response_masks_tensor,
                "rewards": rewards_tensor,
                "loss_mask": loss_masks_tensor,
                "rollout_logprobs": rollout_logprobs_tensor,
                "rollout_staleness": torch.tensor(
                    rollout_staleness if rollout_staleness is not None else [0] * len(response_ids),
                    dtype=torch.int32,
                ),
                "is_last_step": (
                    torch.tensor(trajectory_batch["is_last_step"], dtype=torch.bool)
                    if trajectory_batch.get("is_last_step", None) is not None
                    else None
                ),
            },
        )
        # Attach routed_experts only when present, so the flag-off batch dict has
        # exactly the same keys as today (TensorBatch.__eq__ compares key sets).
        if rollout_routed_experts_tensor is not None:
            training_input["rollout_routed_experts"] = rollout_routed_experts_tensor
        training_input.update(distillation_tensors)
        # Stage B (F5/F4): attach the per-token shaping channel + span tags ONLY
        # when present, so the flag-off batch dict has exactly the same keys as
        # today (TensorBatch.__eq__ compares key sets).
        if token_level_shaping_tensor is not None:
            training_input["token_level_shaping"] = token_level_shaping_tensor
        if response_span_tags_tensor is not None:
            training_input["response_span_tags"] = response_span_tags_tensor
        loop_advantages_tensor = collate_response_token_channel(
            loop_advantages,
            response_masks_tensor,
            dtype=torch.float,
            expected_lengths=[len(response) for response in response_ids],
        )
        if loop_advantages_tensor is not None:
            training_input["loop_advantages"] = loop_advantages_tensor
        training_input.metadata = {"uids": uids}
        # For RLOO-N: pass through exclude_from_baseline flags if present
        if trajectory_batch.get("exclude_from_baseline") is not None:
            training_input.metadata["exclude_from_baseline"] = np.array(
                trajectory_batch["exclude_from_baseline"], dtype=bool
            )
        training_input.metadata["consumed_stop_metrics"] = consumed_stop_metrics(
            trajectory_batch.get("stop_reasons"), len(response_ids)
        )
        # padded response length
        training_input.metadata["response_length"] = response_masks_tensor.shape[1]
        if self.cfg.trainer.step_wise_training:
            assert "trajectory_ids" in trajectory_batch, (
                "Expected `trajectory_ids` in trajectory batch for step wise training"
            )
            training_input.metadata["trajectory_ids"] = [
                trajectory_id.to_string() for trajectory_id in trajectory_batch["trajectory_ids"]
            ]
            training_input.metadata["avg_response_length"] = sum(
                len(sample_response_ids)
                for sample_response_ids, is_last_step in zip(response_ids, trajectory_batch["is_last_step"])
                if is_last_step
            ) / len(response_ids)
        else:
            training_input.metadata["avg_response_length"] = sum(
                len(sample_response_ids) for sample_response_ids in response_ids
            ) / len(response_ids)

        logger.info(f"Number of sequences before padding: {len(training_input['sequences'])}")
        training_input = self.pad_batch(training_input)
        logger.info(f"Number of sequences after padding: {len(training_input['sequences'])}")

        return training_input

    @torch.no_grad()
    def postprocess_trajectory_batch(self, trajectory_batch: TrajectoryBatch, uids: List[str]) -> TrajectoryBatch:
        """
        Converts to per token rewards and computes pass@N.

        In the future algorithm specific reward or loss mask post processing should be done here.
        """
        trajectory_batch_for_metrics = trajectory_batch
        uids_for_metrics = uids
        if self.cfg.trainer.step_wise_training:
            trajectory_batch_for_metrics = defaultdict(list)
            for key in trajectory_batch:
                if isinstance(trajectory_batch[key], list):
                    trajectory_batch_for_metrics[key] = [
                        trajectory_batch[key][i]
                        for i in range(len(trajectory_batch[key]))
                        if trajectory_batch["is_last_step"][i]
                    ]
            uids_for_metrics = [
                uid for uid, is_last_step in zip(uids, trajectory_batch["is_last_step"]) if is_last_step
            ]

        # only use `trajectory_batch_for_metrics` for metrics calculation
        # For step-wise training, we only calculate metrics for the last step of each trajectory
        mean_reward, pass_at_n = get_metrics_from_trajectory_batch(
            trajectory_batch_for_metrics,
            uids_for_metrics,
        )

        # Per-sample scalar rewards for this step, kept for callbacks that need the
        # distribution rather than its mean (PreflightGateCallback reads this). Captured
        # here because rewards are converted to per-token form a few lines below and the
        # scalar form is not recoverable afterwards. Token-level rewards are skipped: the
        # gate is defined on a per-sample scalar.
        step_rewards = trajectory_batch_for_metrics["rewards"]
        self._current_step_rewards = (
            [float(r) for r in step_rewards] if step_rewards and not isinstance(step_rewards[0], list) else []
        )

        # these use the full trajectory batch
        rewards: Union[List[float], List[List[float]]] = trajectory_batch["rewards"]
        responses: List[List[int]] = trajectory_batch["response_ids"]
        per_token_rewards: List[List[float]] = []

        # Check if rewards are already token-level (List[List[float]]) or response-level (List[float])
        if rewards and isinstance(rewards[0], list):
            # Token-level rewards: rewards is List[List[float]]
            per_token_rewards = rewards
        else:
            # Response-level rewards: rewards is List[float], convert to per-token rewards.
            # Zero-token responses keep an empty credit list; see scalar_reward_token_credit.
            per_token_rewards = [
                scalar_reward_token_credit(reward, response) for reward, response in zip(rewards, responses)
            ]

        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt

        reward_metrics = {
            f"reward/avg_pass_at_{n_samples_per_prompt}": pass_at_n,
            "reward/avg_raw_reward": mean_reward,
        }
        # A group whose rewards all tie carries no advantage signal.
        grouped_rewards = defaultdict(list)
        for uid, reward in zip(uids_for_metrics, step_rewards):
            grouped_rewards[uid].append(float(np.sum(reward)))
        if grouped_rewards:
            reward_metrics["reward/informative_group_fraction"] = sum(
                max(values) > min(values) for values in grouped_rewards.values()
            ) / len(grouped_rewards)
        self.all_metrics.update(reward_metrics)
        logger.info(f"reward/avg_pass_at_{n_samples_per_prompt}: {pass_at_n}, reward/avg_raw_reward: {mean_reward}")

        # re-assign reward but now it's per token rewards
        trajectory_batch["rewards"] = per_token_rewards
        return trajectory_batch

    def select_trajectories(
        self, trajectory_batch: TrajectoryBatch, uids: List[str]
    ) -> tuple[TrajectoryBatch, List[str]]:
        """Apply the configured shared selector before scoring or learner conversion."""
        if self.trajectory_selector is None:
            return trajectory_batch, uids
        selection = self.trajectory_selector.select(trajectory_batch, uids)
        self.all_metrics.update(selection.metrics)
        return selection.trajectory_batch, selection.uids

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """Calculate advantages and returns for the data batch.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["response_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["loss_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["rewards"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `.metadata["uids"]`: List[str]

        Adds:
            - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["returns"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        token_level_rewards = data["rewards"]

        if self.cfg.trainer.step_wise_training:
            is_last_step = data["is_last_step"].bool()
            response_mask = data["response_mask"]
            index = np.array(data.metadata["uids"])
            adv_estimator = self.cfg.trainer.algorithm.advantage_estimator
            config = self.cfg.trainer.algorithm
            values = data["values"]
            gamma = self.cfg.trainer.algorithm.gamma
            lambd = self.cfg.trainer.algorithm.lambd
            grpo_norm_by_std = self.cfg.trainer.algorithm.grpo_norm_by_std
            last_step_rewards = token_level_rewards[is_last_step]
            # compatible with any advantage estimator
            last_step_advantages, last_step_returns = compute_advantages_and_returns(
                token_level_rewards=last_step_rewards,
                response_mask=response_mask[is_last_step],
                index=index[is_last_step.cpu().numpy()],
                adv_estimator=adv_estimator,
                values=values[is_last_step] if values is not None else None,
                config=config,
                gamma=gamma,
                lambd=lambd,
                grpo_norm_by_std=grpo_norm_by_std,
                group_advantage_invariant=self.group_advantage_invariant,
            )
            traj_ids = (
                torch.cat([torch.tensor([False], device=is_last_step.device), is_last_step[:-1]]).int().cumsum(dim=0)
            )
            num_groups = traj_ids[-1].item() + 1
            assert num_groups == len(last_step_advantages), (
                f"number of groups {num_groups} doesn't match the number of trajectories as given by `is_last_step` {len(last_step_advantages)}. The `is_last_step` tensor is likely malformed"
            )
            advantages = last_step_advantages[traj_ids]
            returns = last_step_returns[traj_ids]
        else:
            # For RLOO-N: pass exclude_from_baseline if present in metadata
            exclude_from_baseline = data.metadata.get("exclude_from_baseline", None)
            # Stage C (F6): thread the per-token PBS shaping channel into the
            # advantage estimator when it is present. The dispatcher forwards it
            # via **kwargs; only the rloo_n_pbs combiner consumes it (every other
            # estimator ignores the extra kwarg), and when the key is absent
            # (channel off) token_level_shaping is None -> byte-identical path.
            token_level_shaping = data["token_level_shaping"] if "token_level_shaping" in data else None
            advantages, returns = compute_advantages_and_returns(
                token_level_rewards=token_level_rewards,
                response_mask=data["response_mask"],
                index=data.metadata["uids"],
                adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
                config=self.cfg.trainer.algorithm,
                values=data["values"],
                gamma=self.cfg.trainer.algorithm.gamma,
                lambd=self.cfg.trainer.algorithm.lambd,
                grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
                exclude_from_baseline=exclude_from_baseline,
                token_level_shaping=token_level_shaping,
                group_advantage_invariant=self.group_advantage_invariant,
            )
        data["returns"] = returns
        data["advantages"] = advantages

        # remove padding while calculating metrics
        pad_size = data.metadata.get("pad_size", 0)
        num_samples = len(token_level_rewards)

        return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
        if self.cfg.trainer.step_wise_training:
            avg_rewards: float = return_sums[data["is_last_step"][: num_samples - pad_size]].mean().item()
        else:
            avg_rewards: float = return_sums.mean().item()

        avg_response_length = data.metadata["avg_response_length"]
        data = data.to("cpu")

        valid_advantages = torch.masked_select(
            data["advantages"][: num_samples - pad_size, ...], data["response_mask"][: num_samples - pad_size].bool()
        )
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update(
            {
                "avg_final_rewards": avg_rewards,
                "avg_response_length": avg_response_length,
                "avg_advantages": avg_advantages,
                "avg_advantages_abs": avg_advantages_abs,
            }
        )

        logger.info(f"avg_final_rewards: {avg_rewards}, avg_response_length: {avg_response_length}")
        self.all_metrics.update(
            {
                "loss/avg_final_rewards": avg_rewards,
                "loss/avg_raw_advantages": avg_advantages,
                "loss/avg_raw_advantages_abs": avg_advantages_abs,
            }
        )
        return data

    @staticmethod
    def apply_loop_advantages(data: TrainingInputBatch) -> TrainingInputBatch:
        """Add loop credit after normalization, or return unchanged when the channel is absent."""
        loop_advantages = data.get("loop_advantages")
        if loop_advantages is None:
            return data
        advantages = data["advantages"]
        loop_advantages = loop_advantages.to(device=advantages.device, dtype=advantages.dtype)
        data["advantages"] = advantages + loop_advantages * data["response_mask"]
        return data

    def finalize_advantages_for_training(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """Apply configured normalization before finalizing the advantage tensor."""
        if self.cfg.trainer.algorithm.advantage_batch_normalize:
            data = normalize_advantages_dict(data)
        return self.apply_loop_credit_and_drop_advantage_inputs(data)

    def apply_loop_credit_and_drop_advantage_inputs(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """Apply loop credit, then remove rewards, loop_advantages, and uids before worker dispatch."""
        data = self.apply_loop_advantages(data)
        data.pop("rewards")
        data.pop("loop_advantages", None)
        data.metadata.pop("uids")
        return data

    def dump_data(self, data: TrainingInputBatch, file_name: str):
        """
        Dump data to pickle file
        """
        data_save_dir = Path(self.cfg.trainer.export_path) / "dumped_data"
        data_save_dir.mkdir(parents=True, exist_ok=True)
        data.save(data_save_dir / f"{file_name}.pkl")

    def pad_batch(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        """Pad the batch to be divisible by dp size"""
        dp_size = self.policy_model.actor_infos[0].rank.dp_size
        if self.critic_model is not None:
            dp_size = math.lcm(dp_size, self.critic_model.actor_infos[0].rank.dp_size)
        if self.ref_model is not None:
            dp_size = math.lcm(dp_size, self.ref_model.actor_infos[0].rank.dp_size)

        pad_size = math.ceil(training_input.batch_size / dp_size) * dp_size - training_input.batch_size
        new_tensors = {}
        training_input.metadata["pad_size"] = pad_size
        if pad_size == 0:
            return training_input
        for key, tensor in training_input.items():
            if tensor is not None:
                additional_dims = tuple(tensor.shape[1:]) if len(tensor.shape) > 1 else ()

                if key == "is_last_step":
                    padding_tensor = torch.ones(pad_size, *additional_dims, dtype=tensor.dtype, device=tensor.device)
                elif key == "loss_mask":
                    # ensures that padding tensors don't count towards the loss
                    padding_tensor = torch.zeros(pad_size, *additional_dims, dtype=tensor.dtype, device=tensor.device)
                else:
                    # ensures all padding tensors are in a valid format by cloning `pad_size` from the original input
                    # `pad_size` is guaranteed to be smaller than batch_size
                    padding_tensor = tensor[:pad_size].clone()
                new_tensors[key] = torch.cat([tensor, padding_tensor], dim=0)

        new_training_input = TrainingInputBatch(new_tensors)
        new_training_input.metadata = {}
        new_training_input.metadata["uids"] = training_input.metadata["uids"] + [f"pad{i}" for i in range(pad_size)]
        if "trajectory_ids" in training_input.metadata:
            new_training_input.metadata["trajectory_ids"] = training_input.metadata["trajectory_ids"] + [
                f"pad{i}" for i in range(pad_size)
            ]
        for key, value in training_input.metadata.items():
            if key not in ["uids", "trajectory_ids"]:
                # Extend numpy bool arrays so they stay aligned with the padded batch
                if key == "exclude_from_baseline" and isinstance(value, np.ndarray):
                    new_training_input.metadata[key] = np.concatenate([value, np.ones(pad_size, dtype=value.dtype)])
                else:
                    new_training_input.metadata[key] = copy.deepcopy(value)
        return new_training_input

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """
        Calculate values from the critic, log probs from the policy and ref model, and rewards from the reward model
        and then calculate the kl divergence between the action log probs and the base action log probs.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["attention_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `.metadata["response_length"]`: Int

        Adds:
            - `["base_action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        # MoE router-replay (R3): the pre-update old-logprob / ref forward MUST
        # replay the SAME captured routing as the training forward, otherwise the
        # old-logprob pass uses NATIVE top-k routing while the training pass
        # (worker.training_step -> model.forward(rollout_routed_experts=...)) uses
        # REPLAY routing. For an MoE policy the two routings pick different experts
        # -> divergent logprobs over the SAME tokens/weights -> a huge step-1
        # importance ratio (log_ratio_abs_max ~ 19, policy_loss ~ 1e4,
        # raw_grad_norm ~ 1e5) that corrupts the policy. Threading routed_experts
        # into the forward-pass batch makes old/ref/train forwards use the
        # identical (replay) path so step 1 is genuinely on-policy (log_ratio ~ 0).
        # Gated on presence: flag-off (8B / no router-replay) batches never carry
        # this key, so the selected key set is byte-identical to before.
        fwd_keys = ["sequences", "attention_mask"]
        if "rollout_routed_experts" in training_input.keys():
            fwd_keys.append("rollout_routed_experts")
        data_fwd_pass = training_input.select(keys=fwd_keys, metadata_keys=["response_length"])
        data_fwd_pass.metadata["global_step"] = self.global_step

        def collect_results(actor_infos, results, key):
            ret_outputs: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(actor_infos, results)
            return ret_outputs[key]

        base_log_probs = None
        action_log_probs = None
        values = None

        # calculate critic values
        if self.colocate_all and self.critic_model is not None:
            self.critic_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        if self.critic_model is not None:
            value_refs = self.critic_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)
            if self.colocate_all:
                all_rank_values = ray.get(value_refs)
                values = collect_results(self.critic_model.actor_infos, all_rank_values, key="output")
                self.critic_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        # calculate ref log probs
        if self.ref_model is not None:
            if self.cfg.trainer.placement.colocate_policy_ref or self.colocate_all:
                self.ref_model.backload_to_gpu()

            base_action_log_probs_refs = self.ref_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)

        if self.ref_model is not None:
            # handle colocate policy and ref model
            if self.cfg.trainer.placement.colocate_policy_ref or self.colocate_all:
                all_rank_base_log_probs: List[TrainingOutputBatch] = ray.get(base_action_log_probs_refs)
                base_log_probs = collect_results(self.ref_model.actor_infos, all_rank_base_log_probs, key="output")
                self.ref_model.offload_to_cpu()
                ray.get(self.ref_model.async_run_ray_method("pass_through", "empty_cache"))
        else:
            base_log_probs = None

        # calculate action log probs
        if self.colocate_all:
            self.policy_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        action_log_probs_refs = self.policy_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)
        if self.colocate_all:
            all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(action_log_probs_refs)
            action_log_probs = collect_results(self.policy_model.actor_infos, all_rank_action_log_probs, key="output")
            self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        # wait all models done
        # if not colocate_policy_ref, then need to gather base_log_probs
        # if self.critic_model is not None, then need to gather value
        if not self.colocate_all:
            if not self.cfg.trainer.placement.colocate_policy_ref:
                if self.critic_model is not None:
                    all_rank_values = ray.get(value_refs)
                    values = collect_results(self.critic_model.actor_infos, all_rank_values, key="output")

                if self.ref_model is not None:
                    all_rank_base_log_probs: List[TrainingOutputBatch] = ray.get(base_action_log_probs_refs)
                    base_log_probs = collect_results(self.ref_model.actor_infos, all_rank_base_log_probs, key="output")
                else:
                    base_log_probs = None

            elif self.critic_model is not None:
                all_rank_values = ray.get(value_refs)
                values = collect_results(self.critic_model.actor_infos, all_rank_values, key="output")

            all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(action_log_probs_refs)
            action_log_probs = collect_results(self.policy_model.actor_infos, all_rank_action_log_probs, key="output")

        if not self.colocate_all:
            empty_cache_refs = self.policy_model.async_run_ray_method("pass_through", "empty_cache")
            if self.ref_model is not None:
                empty_cache_refs.extend(self.ref_model.async_run_ray_method("pass_through", "empty_cache"))
            if self.critic_model is not None:
                empty_cache_refs.extend(self.critic_model.async_run_ray_method("pass_through", "empty_cache"))
            ray.get(empty_cache_refs)

        sequences_all: torch.Tensor = training_input["sequences"]
        # NOTE (sumanthrh): The slicing is needed to make sure that the batch dimension doesn't change for the tensordict.
        base_log_probs = base_log_probs[: len(sequences_all)] if base_log_probs is not None else None
        action_log_probs = action_log_probs[: len(sequences_all)]
        values = values[: len(sequences_all)] if values is not None else None

        training_input["base_action_log_probs"] = base_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = values

        if self._training_metrics_enabled and training_input.get("rollout_logprobs") is not None:
            self.all_metrics.update(
                mismatch_ratio_metrics(
                    action_log_probs,
                    training_input["rollout_logprobs"],
                    training_input["loss_mask"],
                    training_input["rollout_staleness"],
                    eps_clip_low=self.cfg.trainer.algorithm.eps_clip_low,
                    eps_clip_high=self.cfg.trainer.algorithm.eps_clip_high,
                )
            )

        if self.cfg.generator.sampling_params.logprobs is not None and training_input["rollout_logprobs"] is not None:
            # calculates the difference in probs between inference and trainer components
            # only consider response tokens.
            # NOTE (Fix A-extend, 2026-06-07): rollout_logprobs can be None for a
            # whole batch when use_tis is on but every trajectory lacked logprobs
            # (graceful-degrade path in convert_to_training_input). Subscripting None
            # here is what crashed the 80B R3+TIS train loop at global_step 1
            # ('NoneType' object is not subscriptable). Skip the inference/train prob-diff
            # diagnostic for that batch; the batch still trains as standard (non-TIS) loss.
            logprobs_diff = (
                training_input["rollout_logprobs"][training_input["loss_mask"] > 0]
                - action_log_probs[training_input["loss_mask"] > 0]
            )
            prob_diff = logprobs_diff.exp().abs()
            prob_diff_mean = prob_diff.mean().item()
            prob_diff_std = prob_diff.std().item()
            self.all_metrics.update(
                {
                    "policy/rollout_train_prob_diff_mean": prob_diff_mean,
                    "policy/rollout_train_prob_diff_std": prob_diff_std,
                }
            )
        # Always log KL divergence as a diagnostic, even when not used as penalty
        if base_log_probs is not None:
            _kl = compute_approx_kl(
                action_log_probs,
                base_log_probs,
                loss_mask=training_input["loss_mask"],
                kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
            )
            _kl_mean = masked_mean(_kl, training_input["loss_mask"], dim=-1).mean().item()
            _kl_max = torch.max(_kl.abs(), dim=-1)[0].mean().item()
            self.all_metrics.update(
                {
                    "reward/policy_ref_kl": _kl_mean,
                    "reward/policy_ref_kl_max": _kl_max,
                }
            )

        return training_input

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Applies a penalty for KL divergence between the policy log probs and the base model log probs to the rewards."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        rewards: torch.Tensor = data["rewards"]
        base_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # single batched computation
        kl: Float[torch.Tensor, "batch_size seqlen"] = compute_approx_kl(  # type: ignore
            action_log_probs,
            base_action_log_probs,
            loss_mask=loss_masks_all,
            kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
        )
        kl_max: Float[torch.Tensor, "batch_size"] = torch.max(kl.abs(), dim=-1)[0]  # noqa: F821
        kl_mean: Float[torch.Tensor, "batch_size"] = masked_mean(kl, loss_masks_all, dim=-1)  # noqa: F821

        # NOTE (erictang000): only supporting custom rewards currently
        kl_loss_coef = (
            self.reward_kl_controller.value
            if self.reward_kl_controller is not None
            else self.cfg.trainer.algorithm.kl_loss_coef
        )
        rewards = rewards - kl * max(0, kl_loss_coef)
        data["rewards"] = rewards

        avg_kl: float = kl_mean.mean().item()
        avg_kl_max: float = kl_max.mean().item()

        # update the kl controller
        if self.reward_kl_controller is not None:
            self.reward_kl_controller.update(current=avg_kl, n_steps=kl.shape[0])  # n_steps is just the batch size
        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}

        data.metadata["metrics"].update(
            {
                "avg_kl": avg_kl,
                "avg_kl_max": avg_kl_max,
                "kl_loss_coef": kl_loss_coef,
            }
        )

        self.all_metrics.update(
            {
                "loss/avg_kl": avg_kl,
                "loss/avg_kl_max": avg_kl_max,
                "loss/kl_loss_coef": kl_loss_coef,
            }
        )

        return data

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """
        Run the training step for the policy and critic models (this is overlapped if colocate_all is False).
        """
        data.metadata["global_step"] = self.global_step
        # Plumb the batch's minimum staleness to the worker for StaleClip.
        # Workers treat None as "no signal" and skip damping.
        data.metadata["stale_min"] = self.all_metrics.get("async/staleness_min")
        # ── Global length-unbiased normalizer (seq_mean_token_sum_norm_global) ──
        # Every backend consumes the denominator from batch metadata. Keeping this
        # contract at the driver boundary prevents a worker override from silently
        # bypassing policy-loss semantics and avoids an in-worker collective.
        if self.cfg.trainer.algorithm.loss_reduction == GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION:
            actor_infos = self.policy_model.actor_infos
            ranks_per_dp_group = len(actor_infos) // actor_infos[0].rank.dp_size
            data.metadata[GLOBAL_LOSS_DENOM_METADATA_KEY] = compute_global_loss_denom(
                data["advantages"],
                self.cfg.trainer.algorithm.max_seq_len,
                ranks_per_dp_group,
            )
        if self.colocate_all:
            if self.critic_model is not None:
                with Timer("critic_train", self.all_timings):
                    self.critic_model.backload_to_gpu()
                    critic_refs = self.critic_model.async_run_ray_method("mesh", "ppo_train", data)
                    critic_statuses = collect_actor_results(
                        self.critic_model.actor_infos,
                        critic_refs,
                        operation="critic ppo_train",
                    )
                    self.critic_model.offload_to_cpu()
            with Timer("policy_train", self.all_timings):
                self.policy_model.backload_to_gpu()
                policy_refs = self.policy_model.async_run_ray_method("mesh", "ppo_train", data)
                policy_statuses = collect_actor_results(
                    self.policy_model.actor_infos,
                    policy_refs,
                    operation="policy ppo_train",
                )
        else:
            if self.cfg.trainer.offload_optimizer_during_rollouts:
                with Timer("backload_policy_optimizer_to_gpu", self.all_timings):
                    self.policy_model.backload_to_gpu(backload_optimizer=True, backload_model=False)
            if self.critic_model is not None:
                with Timer("policy_critic_overlap_train", self.all_timings):
                    policy_refs = self.policy_model.async_run_ray_method("mesh", "ppo_train", data)
                    critic_refs = self.critic_model.async_run_ray_method("mesh", "ppo_train", data)
                    all_statuses = collect_actor_results(
                        self.policy_model.actor_infos + self.critic_model.actor_infos,
                        policy_refs + critic_refs,
                        operation="policy and critic ppo_train",
                    )
                    policy_statuses = all_statuses[: len(policy_refs)]
                    critic_statuses = all_statuses[len(policy_refs) :]
            else:
                with Timer("policy_train", self.all_timings):
                    policy_refs = self.policy_model.async_run_ray_method("mesh", "ppo_train", data)
                    policy_statuses = collect_actor_results(
                        self.policy_model.actor_infos,
                        policy_refs,
                        operation="policy ppo_train",
                    )

        empty_cache_refs = []
        if self.critic_model is not None:
            critic_status = critic_statuses[0].metadata["train_status"]
            for k, v in critic_status.items():
                self.all_metrics.update({f"critic/{k}": v})
            empty_cache_refs += self.critic_model.async_run_ray_method("pass_through", "empty_cache")

        policy_status = policy_statuses[0].metadata["train_status"]
        for k, v in policy_status.items():
            self.all_metrics.update({f"policy/{k}": v})
        empty_cache_refs += self.policy_model.async_run_ray_method("pass_through", "empty_cache")
        ray.get(empty_cache_refs)

        return policy_status

    def _get_dp_group_models(self, rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        return model._actor_handlers[rank]

    def _get_mesh_rank(self, rank: int, model_type: str = "") -> MeshRank:
        model: PPORayActorGroup = getattr(self, model_type)
        actor_info: ActorInfo = model.actor_infos[rank]
        return actor_info.rank

    async def save_checkpoints(self) -> None:
        """Save and publish a complete checkpoint before returning."""
        snapshot = await asyncio.to_thread(self._snapshot_checkpoint, await self._rollout_state())
        upload_duration, cleanup_duration = await self._finish_checkpoint_upload(snapshot, commit=True)
        self.all_timings["checkpoint_upload"] = self.all_timings.get("checkpoint_upload", 0.0) + upload_duration
        self.all_timings["cleanup_old_checkpoints"] = (
            self.all_timings.get("cleanup_old_checkpoints", 0.0) + cleanup_duration
        )

    async def _rollout_state(self) -> TrainingContextState:
        """Return the rollout state saved as ``data.pt`` with each checkpoint."""
        return await self.context.state_dict()

    def _restore_rollout_state(self, state: object) -> None:
        """Hold a checkpoint's rollout state until the training context is restored from it."""
        if not isinstance(state, TrainingContextState):
            raise ValueError(f"checkpoint data state is a {type(state).__name__}, not a rollout context state")
        self._restored_rollout_state = state

    def _snapshot_checkpoint(self, rollout_state: TrainingContextState) -> CheckpointSnapshot:
        """
        Stage model shards and serialize trainer state for later publication.

        If colocate_all is True, assumes that the policy model is currently on GPU.
        """
        # Create global step folder structure
        step = self.global_step
        global_step_folder = os.path.join(self.cfg.trainer.ckpt_path, f"global_step_{step}")
        policy_save_dir = os.path.join(global_step_folder, POLICY_CHECKPOINT_SUBDIRECTORY)
        critic_save_dir = os.path.join(global_step_folder, "critic")

        io.makedirs(global_step_folder, exist_ok=True)

        # Save policy checkpoint
        ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through",
                "save_checkpoint",
                ckpt_dir=policy_save_dir,
                tokenizer=self.tokenizer,
            )
        )

        # Save critic checkpoint (if it exists)
        if self.critic_model is not None:
            if self.colocate_all:
                self.policy_model.offload_to_cpu()
                self.critic_model.backload_to_gpu()

            ray.get(
                self.critic_model.async_run_ray_method(
                    "pass_through",
                    "save_checkpoint",
                    ckpt_dir=critic_save_dir,
                    tokenizer=self.tokenizer,
                )
            )

            if self.colocate_all:
                self.critic_model.offload_to_cpu()
                self.policy_model.backload_to_gpu()

        # Serialize rollout data state for publication after the rank uploads complete.
        dataloader_save_path = os.path.join(global_step_folder, "data.pt")
        dataloader_buffer = stdlib_io.BytesIO()
        torch.save(rollout_state, dataloader_buffer)
        dataloader_payload = dataloader_buffer.getvalue()

        # Save additional trainer state
        trainer_state = {
            "global_step": step,
            "config": self.cfg,
            "distillation_scored_tokens_total": self.distillation_scored_tokens_total,
            "domain_gradient_balance_state": self._domain_balancer.state_dict() if self._domain_balancer else None,
        }
        trainer_state_path = os.path.join(global_step_folder, TRAINER_STATE_FILENAME)
        trainer_state_buffer = stdlib_io.BytesIO()
        torch.save(trainer_state, trainer_state_buffer)
        latest_checkpoint_file = os.path.join(self.cfg.trainer.ckpt_path, LATEST_CHECKPOINT_FILE)
        return CheckpointSnapshot(
            step=step,
            upload_started_at=time.monotonic(),
            dataloader_path=dataloader_save_path,
            dataloader_payload=dataloader_payload,
            trainer_state_path=trainer_state_path,
            trainer_state_payload=trainer_state_buffer.getvalue(),
            marker_path=latest_checkpoint_file,
        )

    def _cleanup_old_checkpoints(self):
        max_ckpts = self.cfg.trainer.max_ckpts_to_keep
        # Disabled (the default): keep all checkpoints. The payload is a no-op at
        # this value, so skip the per-node Ray lease entirely. Leasing a fresh
        # worker on every node with hard affinity can fail on a CPU-saturated
        # cluster for no benefit.
        if max_ckpts < 0:
            return

        protected_steps = protected_hf_export_steps(self.cfg.trainer.ckpt_path)

        if not self._node_ids:
            self._node_ids = get_node_ids(self.policy_model, self.critic_model, self.ref_model)
        try:
            run_on_each_node(
                self._node_ids,
                cleanup_old_checkpoints,
                self.cfg.trainer.ckpt_path,
                max_ckpts,
                protected_steps,
            )
        except ray.exceptions.RayError as e:
            # Best-effort: cleanup runs only after a successful checkpoint save,
            # so any per-node dispatch failure -- worker lease failure, worker or
            # node death, or a failure raised inside the remote task -- is logged
            # rather than propagated. The checkpoint is already on disk; cleanup
            # must not kill the run.
            logger.warning(f"Per-node checkpoint cleanup failed, continuing: {e}")

        # Driver-side cleanup. For a shared ckpt_path (GPFS, S3) this alone
        # suffices; the per-node fan-out above only matters for node-local dirs.
        cleanup_old_checkpoints(self.cfg.trainer.ckpt_path, max_ckpts, protected_steps)

    def load_checkpoints(self) -> Tuple[int, str]:
        """
        Load complete checkpoint state and return the global_step to resume from.
        Returns 0 if no checkpoint is loaded.

        If colocate_all is True, assumes that the policy model is currently on GPU.

        Returns:
            global_step: The global step to resume from.
            checkpoint_path: The path to the checkpoint.
        """
        checkpoint_path = None
        # Check if resumption is enabled
        if self.resume_mode == ResumeMode.NONE:
            logger.info("Checkpoint resumption disabled, starting training from scratch")
            return 0, None
        # first, let's get resume_path
        elif self.resume_mode == ResumeMode.LATEST:
            latest_checkpoint_file = os.path.join(self.cfg.trainer.ckpt_path, LATEST_CHECKPOINT_FILE)
            if not io.exists(latest_checkpoint_file):
                logger.warning(
                    f"resume_mode=latest found no checkpoint marker at {latest_checkpoint_file}; "
                    "starting training from global_step 0"
                )
                return 0, None
            with io.open_file(latest_checkpoint_file, "r") as f:
                ckpt_iteration = int(f.read().strip())
            checkpoint_path = os.path.join(self.cfg.trainer.ckpt_path, f"{GLOBAL_STEP_PREFIX}{ckpt_iteration}")
            # Run validation: Make sure ckpt folder is consistent with latest_ckpt_global_step.txt
            validate_consistency_for_latest_checkpoint(
                self.cfg.trainer.ckpt_path,
                ckpt_iteration,
                checkpoint_path,
                latest_checkpoint_file,
                self.cfg.trainer.ckpt_interval,
            )
        else:
            # Get and validate resume path
            checkpoint_path = self.cfg.trainer.resume_path
            if not checkpoint_path:
                raise ValueError("`trainer.resume_path` must be specified when resume_mode is 'from_path'")
            checkpoint_path = checkpoint_path.rstrip("/")

            # Validate that it's a global_step directory
            if GLOBAL_STEP_PREFIX not in os.path.basename(checkpoint_path):
                raise ValueError(
                    f"`trainer.resume_path` must point to a directory whose name starting with {GLOBAL_STEP_PREFIX}, got: {checkpoint_path}"
                )

        # Validate that the path exists
        if not io.exists(str(checkpoint_path)):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Extract global step from checkpoint path
        global_step = extract_step_from_path(checkpoint_path)
        if global_step == -1:
            raise ValueError(f"Checkpoint path {checkpoint_path} is not a valid checkpoint path")
        logger.info(f"Resuming from global_step: {global_step}")

        # Define paths for different checkpoint components
        policy_ckpt_dir = os.path.join(checkpoint_path, POLICY_CHECKPOINT_SUBDIRECTORY)
        critic_ckpt_dir = os.path.join(checkpoint_path, "critic")
        trainer_state_path = os.path.join(checkpoint_path, TRAINER_STATE_FILENAME)
        dataloader_state_path = os.path.join(checkpoint_path, "data.pt")

        # Validate that required checkpoint files exist
        if not io.exists(trainer_state_path):
            raise FileNotFoundError(f"Trainer state file not found: {trainer_state_path}")

        # 1. Load and validate trainer state
        with io.open_file(trainer_state_path, "rb") as f:
            trainer_state = torch.load(f, map_location="cpu", weights_only=False)
        saved_global_step = trainer_state.get("global_step", global_step)
        if self.cfg.trainer.get("reset_distillation_token_count_on_resume", False):
            self.distillation_scored_tokens_total = 0
        else:
            self.distillation_scored_tokens_total = int(trainer_state.get("distillation_scored_tokens_total", 0))
        if self._domain_balancer is not None:
            balance_state = trainer_state.get("domain_gradient_balance_state")
            if balance_state is None:
                raise ValueError("Cannot resume domain-gradient-balanced OPD without its saved anchor state")
            self._domain_balancer.load_state_dict(balance_state)
        logger.info("Successfully loaded trainer state")
        if saved_global_step != global_step:
            logger.warning(f"Global step mismatch: path={global_step}, saved={saved_global_step}. Using path value.")

        # 2. Load dataloader state if requested and available
        if not self.cfg.trainer.restore_dataloader_state:
            logger.info("Dataloader state restoration disabled; starting the configured dataset from the beginning")
        elif io.exists(dataloader_state_path):
            with io.open_file(dataloader_state_path, "rb") as f:
                dataloader_state = torch.load(f, map_location="cpu", weights_only=False)
            self._restore_rollout_state(dataloader_state)
            logger.info("Successfully loaded dataloader state")
        else:
            logger.warning(
                f"No dataloader state found at {dataloader_state_path}. Dataloader will start from beginning."
            )

        # Match the optimizer residency used when disaggregated checkpoints are
        # saved. Megatron initializes restore buffers before reading checkpoint
        # tensors; leaving the optimizer and gradient buffers on GPU can double
        # their peak allocation and OOM before the first rollout.
        self._offload_policy_optimizer(
            getattr(self, "all_startup_timings", {}), timer_label="offload_policy_optimizer_before_checkpoint_load"
        )

        # 3. Load policy checkpoint
        logger.info(f"Loading policy checkpoint from {policy_ckpt_dir}")
        _ = ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through",
                "load_checkpoint",
                ckpt_dir=policy_ckpt_dir,
                load_training_state=True,
            )
        )
        logger.info("Successfully loaded policy checkpoint")

        # 4. Load critic checkpoint if it exists and we have a critic model
        if self.critic_model is not None:
            logger.info(f"Loading critic checkpoint from {critic_ckpt_dir}")
            _ = ray.get(
                self.critic_model.async_run_ray_method(
                    "pass_through",
                    "load_checkpoint",
                    ckpt_dir=critic_ckpt_dir,
                    load_training_state=True,
                )
            )
            logger.info("Successfully loaded critic checkpoint")

        logger.info(f"Successfully loaded complete checkpoint state from global_step_{global_step}")
        if self.cfg.trainer.get("reset_global_step_on_resume", False):
            logger.info("Resetting the trainer step to 0 at the explicit stage boundary")
            global_step = 0
        return global_step, str(checkpoint_path)

    def handle_hf_export(self) -> None:
        """Persist a request for out-of-band policy checkpoint conversion."""
        with Timer("queue_hf_export", self.all_timings):
            self._handle_hf_export()

    def _handle_hf_export(self) -> None:
        checkpoint_path = os.path.join(self.cfg.trainer.ckpt_path, f"{GLOBAL_STEP_PREFIX}{self.global_step}")
        trainer_state_path = os.path.join(checkpoint_path, TRAINER_STATE_FILENAME)
        if not io.exists(trainer_state_path):
            raise RuntimeError(
                f"Cannot request HF export for global_step_{self.global_step}: "
                f"completed checkpoint marker is missing at {trainer_state_path}"
            )

        placement = self.cfg.trainer.placement
        model = self.cfg.trainer.policy.model
        request = HFExportRequest(
            step=self.global_step,
            checkpoint_base_path=self.cfg.trainer.ckpt_path,
            checkpoint_path=checkpoint_path,
            export_path=self.cfg.trainer.export_path,
            model_path=model.path,
            num_nodes=placement.policy_num_nodes,
            gpus_per_node=placement.policy_num_gpus_per_node,
            model_source_uri=model.source_uri,
            model_source_identity=model.source_identity,
            hf_hub_repo_id=self.cfg.trainer.get("hf_hub_repo_id"),
            hf_hub_private=self.cfg.trainer.get("hf_hub_private", False),
            hf_hub_revision=self.cfg.trainer.get("hf_hub_revision", DEFAULT_HF_HUB_REVISION),
            hf_upload_mode=HFUploadMode(self.cfg.trainer.get("hf_upload_mode", DEFAULT_HF_UPLOAD_MODE)),
        )
        existing = read_hf_export_request(checkpoint_path)
        if existing is not None:
            if existing.status is HFExportStatus.COMPLETE:
                logger.info(f"HF export for global_step_{self.global_step} is already complete")
                return
            if existing.status is HFExportStatus.IN_PROGRESS:
                logger.info(f"HF export for global_step_{self.global_step} is already in progress")
                return
            request = replace(
                request,
                attempts=existing.attempts,
                timeout=existing.timeout,
                last_exit_code=existing.last_exit_code,
            )
            logger.info(f"Refreshing pending HF export request for global_step_{self.global_step}")
        request_path = write_hf_export_request(request)
        logger.info(f"Queued out-of-band HF export for global_step_{self.global_step}: {request_path}")

    def _log_startup_timings(self) -> None:
        publish_startup_timings(
            self.all_startup_timings,
            self.all_timings,
            step=self.global_step,
            tracker=self.tracker,
            console=self._log_metrics_stdout,
        )

    def _log_metrics_stdout(self, payload: Dict[str, Any], step: int, kind: str = "train") -> None:
        """Mirror the wandb/tracker payload to stdout so metrics are recoverable without wandb access."""

        def _coerce(v):
            try:
                if isinstance(v, (int, float, bool, str)) or v is None:
                    return v
                if hasattr(v, "item"):
                    return v.item()
                return float(v)
            except Exception:
                return str(v)

        values = {}
        try:
            values = {k: _coerce(v) for k, v in payload.items()}
            serialised = json.dumps(values, sort_keys=True)
        except Exception as e:
            serialised = f'{{"_serialize_error": "{e}"}}'
        logger.info(f"WANDB_MIRROR kind={kind} step={step} metrics={serialised}")
        if self._training_metrics_enabled:
            record_training_metrics(values, step=step, kind=kind)

    def update_ref_with_policy(self):
        """
        Update the reference model with the policy model weights (required by some algorithms)

        If colocate_all is enabled:
        - before calling this method, the policy model should be on GPU, and inference engine should be asleep / on CPU.
        - after calling this method, the same model placement still holds.
        """
        # TODO(tgriggs): Make policy-to-ref sync faster.
        policy_export_dir = policy_export_path(self.cfg.trainer.export_path, self.global_step)
        ray.get(
            self.policy_model.async_run_ray_method("pass_through", "save_hf_model", policy_export_dir, self.tokenizer)
        )
        # NOTE (sumanthrh): This is for the memory efficient case where we can't keep policy and ref model state on GPU together
        # We thus offload the policy model to CPU and then load the ref model from the policy model checkpoint, and then backload the policy model to GPU
        if self.colocate_all:
            self.policy_model.offload_to_cpu()
        ray.get(self.ref_model.async_init_model(policy_export_dir))
        if self.colocate_all:
            self.ref_model.offload_to_cpu()
            self.policy_model.backload_to_gpu()

        # Clean up temporary saved model files
        try:
            shutil.rmtree(policy_export_dir)
            logger.info(f"Cleaned up temporary policy export directory: {policy_export_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary policy export directory {policy_export_dir}: {e}")

        logger.info("Successfully update ref model with policy model, training continue.")
