"""Disposable multi-node Grug benchmark over the routed expert-block schedule.

The Iris training driver can launch this module with ``--entrypoint``. Its Hydra
arguments are retained in the result but do not configure a production trainer.
``++sparse_experiment.mode=smoke`` exercises the same Iris gang, package, Ray,
and output path without allocating a model. The default runs the pinned Grug.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_REPO = "marin-community/grug-67b-a2b-sft-s2-thinking-step630"
MODEL_REVISION = "6808fe5c219471517bd51df35addefd38ebebf89"
MAX_MODEL_LEN = 128


@dataclass(frozen=True)
class Geometry:
    policy_gpus: int
    policy_pp: int
    policy_ep: int
    engines: int
    engine_dp: int
    engine_pp: int

    @property
    def gpus(self) -> int:
        return self.policy_gpus + self.engines * self.engine_dp * self.engine_pp


def _config(model_path: str, geometry: Geometry):
    import hydra
    from skyrl_train.entrypoints.main_base import config_dir
    from skyrl_train.utils.utils import validate_cfg

    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.logger = "console"
    validate_cfg(cfg)
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = False
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = 32
    cfg.trainer.policy_mini_batch_size = 32
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_entropy_loss = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = geometry.policy_gpus // 8
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = geometry.policy_pp
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = geometry.policy_ep
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.policy.megatron_config.ddp_config.overlap_grad_reduce = True
    cfg.trainer.policy.megatron_config.ddp_config.overlap_param_gather = False
    cfg.trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32 = False
    cpu_offload = geometry.policy_gpus == 16
    cfg.trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload = cpu_offload
    cfg.trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction = float(cpu_offload)
    cfg.trainer.policy.optimizer_config.lr = 1.0e-6
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.inference_engine_data_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_expert_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_pipeline_parallel_size = geometry.engine_pp
    cfg.generator.num_inference_engines = geometry.engines
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.gpu_memory_utilization = 0.5
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 1.0
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 4
    cfg.generator.weight_sync_transport = "expert_block"
    cfg.generator.expert_block_sync.verify = False
    cfg.generator.engine_init_timeout_seconds = 1800
    return cfg


def _padded_batch(pad_token_id: int):
    import torch
    from skyrl_train.training_batch import TrainingInputBatch

    batch_size, prompt_length, response_length = 32, 12, 8
    generator = torch.Generator().manual_seed(5)
    body_length = prompt_length + response_length
    total_length = body_length + 6
    sequences = []
    masks = []
    pad_before = (3, 0, 5, 1)
    for row in range(batch_size):
        before = pad_before[row % len(pad_before)]
        body = torch.randint(10, 500, (body_length,), generator=generator).tolist()
        after = total_length - body_length - before
        sequences.append([pad_token_id] * before + body + [pad_token_id] * after)
        masks.append([0] * before + [1] * body_length + [0] * after)
    sequences = torch.tensor(sequences, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    response_mask = attention_mask[:, -response_length:]
    zeros = torch.zeros(batch_size, response_length, dtype=torch.float32)
    advantages = torch.linspace(-1.0, 1.0, response_length).unsqueeze(0).repeat(batch_size, 1)
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "values": zeros.clone(),
            "returns": zeros.clone(),
            "advantages": advantages,
            "loss_mask": response_mask.clone(),
            "response_mask": response_mask.clone(),
        }
    )
    batch.metadata = {"response_length": response_length, "global_step": 0}
    return batch


def _train_step(policy, batch) -> dict:
    import ray

    output = ray.get(policy.async_run_ray_method("pass_through", "ppo_train", batch))[0]
    status = output.metadata["train_status"]
    if not math.isfinite(status["policy_loss"]):
        raise RuntimeError(f"Non-finite policy loss: {status['policy_loss']}")
    return status


def _engine_client(cfg, model_path: str, geometry: Geometry):
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    engines = create_ray_wrapped_inference_engines(
        num_inference_engines=geometry.engines,
        tensor_parallel_size=1,
        pipeline_parallel_size=geometry.engine_pp,
        data_parallel_size=geometry.engine_dp,
        expert_parallel_size=geometry.engine_dp,
        model_dtype="bfloat16",
        pretrain=model_path,
        seed=23,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        engine_init_timeout_seconds=cfg.generator.engine_init_timeout_seconds,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        async_engine=True,
        max_num_batched_tokens=MAX_MODEL_LEN * geometry.engine_dp,
        max_num_seqs=cfg.trainer.train_batch_size,
        tokenizer=tokenizer,
        backend="vllm",
        engine_init_kwargs={"max_model_len": MAX_MODEL_LEN},
    )
    return InferenceEngineClient(engines, tokenizer, cfg)


def _init_policy(cfg, geometry: Geometry):
    import ray
    from ray.util.placement_group import placement_group
    from skyrl_train.utils import get_ray_pg_ready_with_timeout
    from skyrl_train.workers.megatron.megatron_worker import PolicyWorker
    from skyrl_train.workers.worker import PPORayActorGroup

    policy_nodes = geometry.policy_gpus // 8
    bundles = [{"GPU": 8, "CPU": 8} for _ in range(policy_nodes)]
    pg = placement_group(bundles, strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=300)
    policy = PPORayActorGroup(
        cfg,
        num_nodes=policy_nodes,
        num_gpus_per_node=8,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    ray.get(policy.async_init_model(cfg.trainer.policy.model.path))
    return policy


def _output_path() -> Path:
    directory = Path(os.environ.get("IRIS_OUTPUT_DIR", "/tmp"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "sparse-expert-real.json"


def _metadata() -> dict:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "uninstalled"

    def command(*argv: str) -> str:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else f"exit={result.returncode}: {result.stderr[-500:]}"

    return {
        "argv": sys.argv,
        "source_commit": command("git", "rev-parse", "HEAD"),
        "torch_version": version("torch"),
        "vllm_version": version("vllm"),
        "megatron_core_version": version("megatron-core"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ray_address": os.environ.get("RAY_ADDRESS"),
        "iris_task_id": os.environ.get("IRIS_TASK_ID"),
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
    }


def _run_real(records: dict) -> None:
    import ray
    from huggingface_hub import hf_hub_download
    from skyrl_train.inference_engines.base import InferenceEngineInput
    from skyrl_train.utils import initialize_ray
    from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync
    from skyrl_train.weight_sync.expert_block.schedule import to_wire
    from transformers import AutoTokenizer

    records["stage"] = "resolve_staged_model"
    model_path = str(
        Path(hf_hub_download(MODEL_REPO, "config.json", revision=MODEL_REVISION, local_files_only=True)).parent
    )
    records["model"] = {"repo": MODEL_REPO, "revision": MODEL_REVISION, "node_local_snapshot": model_path}
    geometry = Geometry(
        policy_gpus=8 * records["policy_nodes"], policy_pp=2, policy_ep=8, engines=1, engine_dp=8, engine_pp=1
    )
    cfg = _config(model_path, geometry)
    records["geometry"] = asdict(geometry)
    records["config"] = {
        "train_batch_size": cfg.trainer.train_batch_size,
        "micro_train_batch_size_per_gpu": cfg.trainer.micro_train_batch_size_per_gpu,
        "learning_rate": cfg.trainer.policy.optimizer_config.lr,
        "vllm_gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
        "optimizer_cpu_offload": cfg.trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload,
        "overlap_param_gather": cfg.trainer.policy.megatron_config.ddp_config.overlap_param_gather,
    }
    records["stage"] = "initialize_ray"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    initialize_ray(cfg)
    records["ray_nodes"] = [
        {"node_id": node["NodeID"], "node_ip": node["NodeManagerAddress"], "resources": node["Resources"]}
        for node in ray.nodes()
        if node["Alive"]
    ]
    if sum(node["resources"].get("GPU", 0) for node in records["ray_nodes"]) < geometry.gpus:
        raise RuntimeError(f"The Ray gang has fewer than {geometry.gpus} GPUs")
    sync = None
    try:
        records["stage"] = "initialize_receiver_and_policy"
        client = _engine_client(cfg, model_path, geometry)
        policy = _init_policy(cfg, geometry)
        sync = ExpertBlockSync(policy_model=policy, inference_engine_client=client, timeout_seconds=900)

        async def prepare():
            records["source_inventory"] = await sync._policy("inventory")
            records["receiver_inventory"] = await sync._receivers("inventory")
            records["prepare"] = await sync.prepare()
            records["schedule"] = to_wire(sync.schedule)
            await client.pause_generation()
            records["initial_dense"] = asdict(await sync.sync(0))
            records["initial_verify"] = await sync.verify(0)
            records["sender_baselines"] = await sync._policy("experiment_capture_baseline")
            records["receiver_baselines"] = await sync._receivers("experiment_capture_baseline")
            await client.resume_generation()

        records["stage"] = "prepare_initial_dense"
        asyncio.run(prepare())
        # A fourth generated token varied once between byte-identical installs in job h.
        # The first three were stable across its nine completed trials.
        records["token_probe"] = {
            "prompt_token_ids": [[1, 17, 29, 5, 11, 3]],
            "temperature": 0.0,
            "max_tokens": 3,
            "ignore_eos": True,
        }
        score_input = InferenceEngineInput(
            prompt_token_ids=records["token_probe"]["prompt_token_ids"],
            sampling_params={key: records["token_probe"][key] for key in ("temperature", "max_tokens", "ignore_eos")},
        )
        for version in (1, 2):
            records["stage"] = f"train_update_{version}"
            batch = _padded_batch(pad_token_id)
            batch.metadata["global_step"] = version
            train_started = time.perf_counter()
            status = _train_step(policy, batch)
            update = {
                "version": version,
                "train_seconds": time.perf_counter() - train_started,
                "train_status": status,
                "trials": [],
            }
            records["updates"].append(update)
            update["distribution"] = asyncio.run(sync._policy("experiment_distribution", {"version": version}))
            expected_tokens = None
            if records["leading_repeats"]:
                pair = ("dense", "indices_bucket") if version == 1 else ("indices_bucket", "dense")
                order = pair * records["leading_repeats"]
            else:
                order = (
                    ("dense", "indices", "bitmap", "indices_bucket", "bitmap_bucket")
                    if version == 1
                    else ("bitmap_bucket", "indices_bucket", "bitmap", "indices", "dense")
                )
            for index, encoding in enumerate(order):
                records["stage"] = f"install_update_{version}_{encoding}"

                async def trial(index=index, encoding=encoding, version=version):
                    if index:
                        await client.pause_generation()
                        await sync._receivers("experiment_restore_baseline")
                        await client.resume_generation()
                    if encoding == "dense":
                        policy_memory, receiver_memory = await asyncio.gather(
                            sync._policy("experiment_memory", {"reset_peak": True}),
                            sync._receivers("experiment_memory", {"reset_peak": True}),
                        )
                        memory_before = {"policy": policy_memory, "receivers": receiver_memory}
                    start = time.perf_counter()
                    await client.pause_generation()
                    paused = time.perf_counter()
                    if encoding == "dense":
                        detail = {"install": asdict(await sync.sync(version)), "memory_before": memory_before}
                    else:
                        transfer = {"version": version, "encoding": encoding}
                        policy_rows, receiver_rows = await asyncio.gather(
                            sync._policy("experiment_send", transfer), sync._receivers("experiment_receive", transfer)
                        )
                        if {row["participant"] for row in policy_rows} != set(range(sync.schedule.trainer_count)):
                            raise RuntimeError("A trainer did not acknowledge sparse publication")
                        if {row["participant"] for row in receiver_rows} != set(dict(sync.schedule.receiver_bytes)):
                            raise RuntimeError("A receiver did not acknowledge sparse publication")
                        if any(
                            row["version"] != version or row["encoding"] != encoding
                            for row in [*policy_rows, *receiver_rows]
                        ):
                            raise RuntimeError("A participant acknowledged another sparse publication")
                        detail = {"policy": policy_rows, "receivers": receiver_rows}
                    installed = time.perf_counter()
                    await client.resume_generation()
                    published = time.perf_counter()
                    if encoding == "dense":
                        policy_memory, receiver_memory = await asyncio.gather(
                            sync._policy("experiment_memory", {"reset_peak": False}),
                            sync._receivers("experiment_memory", {"reset_peak": False}),
                        )
                        detail["memory_after"] = {"policy": policy_memory, "receivers": receiver_memory}
                    await client.pause_generation()
                    # The transfer must exactly match its scheduled roots. Record any training
                    # replica drift separately so it cannot hide a receiver byte mismatch.
                    verification = await sync.verify(version, require_identical_replicas=False)
                    await client.resume_generation()
                    tokens = (await client.generate(score_input))["response_ids"]
                    return {
                        "encoding": encoding,
                        "pause_seconds": paused - start,
                        "install_seconds": installed - paused,
                        "resume_seconds": published - installed,
                        "publication_seconds": published - start,
                        "detail": detail,
                        "verification": verification,
                        "tokens": tokens,
                    }

                result = asyncio.run(trial())
                update["trials"].append(result)
                _output_path().write_text(json.dumps(records, indent=2, sort_keys=True, default=str))
                if expected_tokens is None:
                    expected_tokens = result["tokens"]
                if result["tokens"] != expected_tokens:
                    raise RuntimeError(f"Tokens disagree across encodings for update {version}")

            async def advance():
                return await asyncio.gather(
                    sync._policy("experiment_advance_baseline"), sync._receivers("experiment_advance_baseline")
                )

            update["advanced_baselines"] = asyncio.run(advance())
            print(f"SPARSE_EXPERT_REAL_UPDATE_PASS version={version} byte_equal=true token_equal=true", flush=True)
        records["complete"] = True
        records["stage"] = "complete"
    finally:
        if sync is not None:
            asyncio.run(sync.close())
        ray.shutdown()


def main() -> None:
    import ray

    mode = "real"
    leading_repeats = 0
    policy_nodes = 2
    for arg in sys.argv[1:]:
        if arg.startswith("++sparse_experiment.mode="):
            mode = arg.split("=", 1)[1]
        if arg.startswith("++sparse_experiment.leading_repeats="):
            leading_repeats = int(arg.split("=", 1)[1])
        if arg.startswith("++sparse_experiment.policy_nodes="):
            policy_nodes = int(arg.split("=", 1)[1])
    if leading_repeats < 0 or leading_repeats > 10:
        raise ValueError("leading_repeats must be between zero and ten")
    if policy_nodes not in (2, 4):
        raise ValueError("policy_nodes must be two or four")
    records = {
        "schema_version": 1,
        "mode": mode,
        "leading_repeats": leading_repeats,
        "policy_nodes": policy_nodes,
        "metadata": _metadata(),
        "updates": [],
        "complete": False,
    }
    try:
        if mode == "smoke":
            ray.init(address=os.environ["RAY_ADDRESS"])
            records["ray_nodes"] = [
                {"node_id": node["NodeID"], "node_ip": node["NodeManagerAddress"], "resources": node["Resources"]}
                for node in ray.nodes()
                if node["Alive"]
            ]
            if len(records["ray_nodes"]) < 2:
                raise RuntimeError("Multi-node smoke did not join two Ray nodes")
            records["complete"] = True
            print("SPARSE_EXPERT_MULTINODE_SMOKE_PASS", flush=True)
            ray.shutdown()
        elif mode == "real":
            _run_real(records)
        else:
            raise ValueError(f"Unknown experiment mode: {mode}")
    except BaseException as error:
        records["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        _output_path().write_text(json.dumps(records, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
