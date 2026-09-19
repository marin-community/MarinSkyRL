"""Opt-in Grug experiment: dense and exact sparse updates over routed expert-block sync.

Run explicitly on one Hopper node. Every candidate starts from the same receiver image,
uses the real #689 source and destination views, and is checked with #690's independent
dense replay outside the publication timer. The trainer baseline advances only after all
receivers acknowledge, replay passes, and generation resumes.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import ray
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.utils import initialize_ray
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync
from skyrl_train.weight_sync.expert_block.schedule import to_wire
from transformers import AutoTokenizer

from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.test_expert_block_sync import GEOMETRIES, TIMEOUT_SECONDS, engine_client
from tests.gpu.test_grug_megatron import _config, _init_policy, _padded_batch, _train_step, _write_tiny_checkpoint


async def _sparse_install(sync: ExpertBlockSync, version: int, encoding: str) -> dict:
    update = {"version": version, "encoding": encoding}
    policy, receivers = await asyncio.gather(
        sync._policy("experiment_send", update), sync._receivers("experiment_receive", update)
    )
    expected_policy = set(range(sync.schedule.trainer_count))
    expected_receivers = set(dict(sync.schedule.receiver_bytes))
    if {row["participant"] for row in policy} != expected_policy:
        raise RuntimeError("A trainer rank did not report the sparse transfer")
    if {row["participant"] for row in receivers} != expected_receivers:
        raise RuntimeError("A receiver rank did not report the sparse transfer")
    if any(row["version"] != version or row["encoding"] != encoding for row in [*policy, *receivers]):
        raise RuntimeError("A sparse participant reported another version or encoding")
    return {"policy": policy, "receivers": receivers}


@pytest.mark.vllm
def test_tiny_grug_sparse_experiment_two_updates(tmp_path):
    geometry = GEOMETRIES["equal-ep"]
    require_hoppers(geometry.gpus)
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=geometry.policy_gpus, pp=geometry.policy_pp, ep=geometry.policy_ep)
    cfg.generator.num_inference_engines = geometry.engines
    cfg.generator.inference_engine_data_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_expert_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_pipeline_parallel_size = geometry.engine_pp
    cfg.generator.weight_sync_transport = "expert_block"
    cfg.generator.expert_block_sync.verify = False
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    initialize_ray(cfg)
    records = {"geometry": asdict(geometry), "updates": [], "complete": False}
    output_dir = Path(os.environ.get("IRIS_OUTPUT_DIR", tmp_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    sync = None
    try:
        client = engine_client(cfg, str(model_path), geometry)
        policy = _init_policy(cfg, geometry.policy_gpus)
        sync = ExpertBlockSync(policy_model=policy, inference_engine_client=client, timeout_seconds=TIMEOUT_SECONDS)

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
            batch = _padded_batch(tokenizer.pad_token_id)
            batch.metadata["global_step"] = version
            train_status = _train_step(policy, batch)
            update_record = {"version": version, "train_status": train_status, "trials": []}
            records["updates"].append(update_record)

            async def distribution(version=version):
                return await sync._policy("experiment_distribution", {"version": version})

            update_record["distribution"] = asyncio.run(distribution())
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
                    started = time.perf_counter()
                    await client.pause_generation()
                    paused = time.perf_counter()
                    if encoding == "dense":
                        detail = asdict(await sync.sync(version))
                    else:
                        detail = await _sparse_install(sync, version, encoding)
                    installed = time.perf_counter()
                    await client.resume_generation()
                    published = time.perf_counter()
                    # The replay broadcasts full trainer weights into scratch and compares all
                    # receiver bytes. Its cost is excluded from the publication timer.
                    await client.pause_generation()
                    verification = await sync.verify(version)
                    await client.resume_generation()
                    tokens = (await client.generate(score_input))["response_ids"]
                    return {
                        "encoding": encoding,
                        "pause_seconds": paused - started,
                        "install_seconds": installed - paused,
                        "resume_seconds": published - installed,
                        "publication_seconds": published - started,
                        "detail": detail,
                        "verification": verification,
                        "tokens": tokens,
                    }

                result = asyncio.run(trial())
                update_record["trials"].append(result)
                if expected_tokens is None:
                    expected_tokens = result["tokens"]
                assert result["tokens"] == expected_tokens

            async def advance():
                return await asyncio.gather(
                    sync._policy("experiment_advance_baseline"),
                    sync._receivers("experiment_advance_baseline"),
                )

            update_record["advanced_baselines"] = asyncio.run(advance())
        records["complete"] = True
        print("SPARSE_EXPERT_TINY_PASS updates=2 candidates=dense,indices,bitmap byte_equal=true token_equal=true")
    finally:
        (output_dir / "sparse-expert-tiny.json").write_text(json.dumps(records, indent=2, sort_keys=True, default=str))
        if sync is not None:
            asyncio.run(sync.close())
        ray.shutdown()
