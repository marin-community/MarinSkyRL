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
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

MODEL_REPO = "marin-community/grug-67b-a2b-sft-s2-thinking-step630"
MODEL_REVISION = "6808fe5c219471517bd51df35addefd38ebebf89"


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
    from huggingface_hub import snapshot_download
    from skyrl_train.inference_engines.base import InferenceEngineInput
    from skyrl_train.utils import initialize_ray
    from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync
    from skyrl_train.weight_sync.expert_block.schedule import to_wire
    from tests.gpu.test_expert_block_sync import Geometry, engine_client
    from tests.gpu.test_grug_megatron import _config, _padded_batch, _train_step
    from tests.gpu.utils import init_worker_with_type
    from transformers import AutoTokenizer

    model_path = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
    records["model"] = {"repo": MODEL_REPO, "revision": MODEL_REVISION, "node_local_snapshot": model_path}
    geometry = Geometry(policy_gpus=16, policy_pp=2, policy_ep=8, engines=1, engine_dp=8, engine_pp=1)
    cfg = _config(model_path, world_size=8, pp=geometry.policy_pp, ep=geometry.policy_ep)
    cfg.trainer.placement.policy_num_nodes = 2
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.policy.optimizer_config.lr = 1.0e-6
    cfg.trainer.policy.optimizer_config.max_grad_norm = 1.0
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.generator.num_inference_engines = geometry.engines
    cfg.generator.inference_engine_data_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_expert_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_pipeline_parallel_size = geometry.engine_pp
    cfg.generator.gpu_memory_utilization = 0.5
    cfg.generator.weight_sync_transport = "expert_block"
    cfg.generator.expert_block_sync.verify = False
    cfg.generator.engine_init_timeout_seconds = 1800
    records["geometry"] = asdict(geometry)
    records["config"] = {
        "train_batch_size": cfg.trainer.train_batch_size,
        "micro_train_batch_size_per_gpu": cfg.trainer.micro_train_batch_size_per_gpu,
        "learning_rate": cfg.trainer.policy.optimizer_config.lr,
        "vllm_gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    initialize_ray(cfg)
    records["ray_nodes"] = [
        {"node_id": node["NodeID"], "node_ip": node["NodeManagerAddress"], "resources": node["Resources"]}
        for node in ray.nodes()
        if node["Alive"]
    ]
    if sum(node["resources"].get("GPU", 0) for node in records["ray_nodes"]) < geometry.gpus:
        raise RuntimeError("The Ray gang has fewer than 24 GPUs")
    sync = None
    try:
        client = engine_client(cfg, model_path, geometry)
        policy = init_worker_with_type(
            "policy", shared_pg=None, colocate_all=False, num_gpus_per_node=8, num_nodes=2, cfg=cfg
        )
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

        asyncio.run(prepare())
        score_input = InferenceEngineInput(
            prompt_token_ids=[[1, 17, 29, 5, 11, 3]],
            sampling_params={"temperature": 0.0, "max_tokens": 4, "ignore_eos": True},
        )
        for version in (1, 2):
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
            order = (
                ("dense", "indices", "bitmap", "indices_bucket", "bitmap_bucket")
                if version == 1
                else ("bitmap_bucket", "indices_bucket", "bitmap", "indices", "dense")
            )
            for index, encoding in enumerate(order):

                async def trial(index=index, encoding=encoding, version=version):
                    if index:
                        await client.pause_generation()
                        await sync._receivers("experiment_restore_baseline")
                        await client.resume_generation()
                    start = time.perf_counter()
                    await client.pause_generation()
                    paused = time.perf_counter()
                    if encoding == "dense":
                        detail = asdict(await sync.sync(version))
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
                    await client.pause_generation()
                    verification = await sync.verify(version)
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
    finally:
        if sync is not None:
            asyncio.run(sync.close())
        ray.shutdown()


def main() -> None:
    import ray

    mode = "real"
    for arg in sys.argv[1:]:
        if arg.startswith("++sparse_experiment.mode="):
            mode = arg.split("=", 1)[1]
    records = {"schema_version": 1, "mode": mode, "metadata": _metadata(), "updates": [], "complete": False}
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
    finally:
        _output_path().write_text(json.dumps(records, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
