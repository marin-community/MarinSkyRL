"""Expert-block weight sync on one Hopper node, for four geometries.

Each case starts a tiny Grug Megatron policy and vLLM engines, trains one PPO step, syncs with
``weight_sync_transport=expert_block`` and verifies the sync. It then reads engine weights back
and compares them, byte for byte, with the trainer's exported weights. It flips one installed
byte on one worker and checks that verification counts exactly that byte. Then it trains a
second step and syncs again.

Opt-in; needs at most six Hopper GPUs. The test's engine actor is defined in this module, so Ray
workers need ``skyrl-train`` on ``PYTHONPATH``. The Grug gate wrapper sets it:

    uv run --frozen --extra vllm --extra megatron --group dev \\
        python marinskyrl/environment_contract.py run-grug-gpu-gate "$PWD" -- \\
        python -m pytest -s skyrl-train/tests/gpu/test_expert_block_sync.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.utils import initialize_ray
from skyrl_train.weight_sync.expert_block.driver import ExpertBlockSync
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_serving import (
    LM_HEAD_NAME,
    MAX_MODEL_LEN,
    ROUTER_NAME,
    assert_engine_weights,
    rank0_validation_snapshot,
)
from tests.gpu.test_grug_megatron import (
    ATTN_GATE_NAME,
    BIAS_NAMES,
    GATED_NORM_NAME,
    PARAMETER_NAMES,
    SERVING_EXPERT_INDEX_BY_NAME,
    _config,
    _init_policy,
    _padded_batch,
    _train_step,
    _write_tiny_checkpoint,
)

# Tensors the transport slices out of fused trainer parameters (interleaved QKV, [gate;up]). The
# trainer snapshot exports them with Megatron-Bridge, which shares no code with the transport.
SLICED_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.2.shared_expert.gate_proj.weight",
    "model.layers.2.shared_expert.up_proj.weight",
]
SYNC_NAMES = [
    *BIAS_NAMES,
    *SERVING_EXPERT_INDEX_BY_NAME,
    LM_HEAD_NAME,
    ROUTER_NAME,
    ATTN_GATE_NAME,
    GATED_NORM_NAME,
    *SLICED_NAMES,
]
TIMEOUT_SECONDS = 180


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


GEOMETRIES = {
    "equal-ep": Geometry(policy_gpus=2, policy_pp=1, policy_ep=2, engines=1, engine_dp=2, engine_pp=1),
    "two-replicas": Geometry(policy_gpus=2, policy_pp=1, policy_ep=2, engines=2, engine_dp=2, engine_pp=1),
    "unequal-ep": Geometry(policy_gpus=2, policy_pp=2, policy_ep=1, engines=1, engine_dp=2, engine_pp=1),
    "receiver-pp2": Geometry(policy_gpus=2, policy_pp=1, policy_ep=2, engines=1, engine_dp=2, engine_pp=2),
}


def flip_one_installed_byte(worker) -> int:
    """Worker RPC for this test: flip byte 1 of the first expert slot of layer 0."""
    for name, parameter in worker.model_runner.model.named_parameters():
        if name.endswith("layers.0.mlp.experts.routed_experts.w13_weight"):
            with torch.no_grad():
                parameter.data.view(-1).view(torch.uint8)[1] ^= 0xFF
            return 1
    return 0


def engine_client(cfg, model_path: str, geometry: Geometry) -> InferenceEngineClient:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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
        enforce_eager=False,
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


@pytest.mark.vllm
@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_expert_block_sync_installs_every_byte_and_verification_catches_a_flipped_one(tmp_path, name, monkeypatch):
    geometry = GEOMETRIES[name]
    require_hoppers(geometry.gpus)
    from skyrl_train.inference_engines.vllm import vllm_engine  # noqa: PLC0415 - vLLM is a GPU-only extra

    class CorruptibleEngine(vllm_engine.AsyncVLLMInferenceEngine):
        async def flip_installed_byte(self):
            return await self._get_engine().collective_rpc(flip_one_installed_byte)

    # The test's actor only adds the byte-flipping RPC. The transport is the production code.
    monkeypatch.setattr(vllm_engine, "AsyncVLLMRayActor", ray.remote(CorruptibleEngine))
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_checkpoint(model_path)
    cfg = _config(str(model_path), world_size=geometry.policy_gpus, pp=geometry.policy_pp, ep=geometry.policy_ep)
    cfg.generator.num_inference_engines = geometry.engines
    cfg.generator.inference_engine_data_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_expert_parallel_size = geometry.engine_dp
    cfg.generator.inference_engine_pipeline_parallel_size = geometry.engine_pp
    cfg.generator.weight_sync_transport = "expert_block"
    cfg.generator.expert_block_sync.verify = True
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    initialize_ray(cfg)
    try:
        client = engine_client(cfg, str(model_path), geometry)
        policy = _init_policy(cfg, geometry.policy_gpus)
        names = [*PARAMETER_NAMES, *BIAS_NAMES, GATED_NORM_NAME, *SLICED_NAMES]
        before = rank0_validation_snapshot(policy, names)
        batch = _padded_batch(tokenizer.pad_token_id)
        batch.metadata["global_step"] = 1
        _train_step(policy, batch)
        trained = rank0_validation_snapshot(policy, names)
        assert any(not torch.equal(trained[key], before[key]) for key in PARAMETER_NAMES)

        sync = ExpertBlockSync(policy_model=policy, inference_engine_client=client, timeout_seconds=TIMEOUT_SECONDS)
        timings = {}

        async def prepare_and_sync(version):
            if not timings:
                timings["prepare"] = await sync.prepare()
                timings["free_bytes_after_prepare"] = [
                    torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())
                ]
            await client.pause_generation()
            started = time.perf_counter()
            install = await sync.sync(version)
            timings[f"install_{version}"] = install
            timings[f"verify_{version}"] = await sync.verify(version)
            timings[f"paused_{version}"] = time.perf_counter() - started
            await client.resume_generation()

        asyncio.run(prepare_and_sync(1))
        assert_engine_weights(client, SYNC_NAMES, trained, BIAS_NAMES, SERVING_EXPERT_INDEX_BY_NAME)

        async def corrupt_and_verify():
            flipped = await client.engines[-1].inference_engine_actor.flip_installed_byte.remote()
            assert sum(flipped) == 1
            await client.pause_generation()
            try:
                with pytest.raises(RuntimeError, match=r"1 of \d+ replayed bytes differ"):
                    await sync.verify(1)
            finally:
                await client.resume_generation()

        asyncio.run(corrupt_and_verify())

        second = _padded_batch(tokenizer.pad_token_id)
        second.metadata["global_step"] = 2
        _train_step(policy, second)
        trained_again = rank0_validation_snapshot(policy, names)
        assert any(not torch.equal(trained_again[key], trained[key]) for key in PARAMETER_NAMES)
        asyncio.run(prepare_and_sync(2))
        assert_engine_weights(client, SYNC_NAMES, trained_again, BIAS_NAMES, SERVING_EXPERT_INDEX_BY_NAME)
        asyncio.run(sync.close())
        print(
            f"EXPERT_BLOCK_VERIFY_PASS geometry={name} policy_gpus={geometry.policy_gpus} policy_pp={geometry.policy_pp} "
            f"policy_ep={geometry.policy_ep} engines={geometry.engines} "
            f"engine_dp={geometry.engine_dp} engine_pp={geometry.engine_pp} syncs=2 byte_equal=true "
            f"corruption_rejected=true prepare_seconds={timings['prepare']} "
            f"install_seconds={[timings[f'install_{v}'].install_seconds for v in (1, 2)]} "
            f"verify_seconds={[timings[f'verify_{v}']['verify_seconds'] for v in (1, 2)]} "
            f"paused_seconds={[timings[f'paused_{v}'] for v in (1, 2)]} "
            f"free_bytes_after_prepare={timings['free_bytes_after_prepare']}",
            flush=True,
        )
    finally:
        ray.shutdown()
