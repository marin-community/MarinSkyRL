"""
Tests for expert parallel (EP).

uv run --isolated --extra dev --extra vllm pytest tests/gpu/test_expert_parallel_inference.py

"""

import asyncio
import pytest
import ray
from typing import Optional
from ray.util.placement_group import PlacementGroup
from omegaconf import DictConfig
from transformers import AutoTokenizer

from tests.gpu.utils import (
    get_available_gpus,
    get_test_prompts,
    init_worker_with_type,
    are_responses_similar,
    get_test_actor_config,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from ray.util.placement_group import placement_group


MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
NUM_GPUS = 4  # Should be divisible by 2


def _check_gpus(num_gpus: int):
    available = get_available_gpus()
    if len(available) < num_gpus:
        pytest.skip(f"Expert parallel tests require >= {num_gpus} GPUs, found {len(available)}: {available}")


def _get_test_cfg() -> DictConfig:
    cfg = get_test_actor_config()

    # Use MoE policy model
    cfg.trainer.policy.model.path = MODEL

    # vLLM generator with EP enabled
    cfg.generator.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.num_inference_engines = NUM_GPUS // 2
    cfg.generator.inference_engine_tensor_parallel_size = 2
    cfg.generator.inference_engine_expert_parallel_size = 2
    cfg.generator.inference_engine_data_parallel_size = 1
    cfg.generator.gpu_memory_utilization = 0.8

    # Small lengths for faster tests
    cfg.generator.max_input_length = 2048
    cfg.generator.sampling_params.max_generate_length = 512

    # Training knobs for tests
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.train_batch_size = 128
    cfg.trainer.policy_mini_batch_size = 128
    cfg.trainer.micro_forward_batch_size_per_gpu = 8
    cfg.trainer.micro_train_batch_size_per_gpu = 8
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    # Small micro batches to fit the MoE in 2 GPUs during training.
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1

    return cfg


async def _run_single_generation(client: InferenceEngineClient, prompts, sampling_params):
    tasks = [client.generate(InferenceEngineInput(prompts=[p], sampling_params=sampling_params)) for p in prompts]
    results = await asyncio.gather(*tasks)
    responses, reasons = [], []
    for r in results:
        responses.extend(r["responses"])
        reasons.extend(r["stop_reasons"])
    return responses, reasons


def init_ray_inference_engines(
    backend: str, tp_size: int, shared_pg: Optional[PlacementGroup], config: DictConfig
) -> InferenceEngineClient:
    """Initialize ray-wrapped inference engines for the specified backend"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    engine = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=tp_size,
        expert_parallel_size=config.generator.inference_engine_expert_parallel_size,
        model_dtype="bfloat16",
        pretrain=MODEL,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        shared_pg=shared_pg,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=False,
        async_engine=True,
        max_num_batched_tokens=8192,
        max_num_seqs=1024,
        tokenizer=tokenizer,
        backend=backend,
    )
    client = InferenceEngineClient(engine, tokenizer, config)
    return client


def test_ep_generation():
    """
    Ensure vLLM generation with expert parallel enabled (EP=2) runs without errors.
    Validate that the number of outputs matches the number of inputs.
    """
    _check_gpus(num_gpus=NUM_GPUS)

    try:
        cfg = _get_test_cfg()
        # Deterministic sampling for stable execution
        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1
        initialize_ray(cfg)

        client = init_ray_inference_engines(
            backend=cfg.generator.backend,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            shared_pg=None,
            config=cfg,
        )

        prompts = get_test_prompts(MODEL, num_samples=4)
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)

        responses, reasons = asyncio.run(_run_single_generation(client, prompts, sampling_params))
        assert len(responses) == len(prompts)
        assert len(reasons) == len(prompts)
    finally:
        ray.shutdown()


def _run_ep_weight_sync(grouped_ep: bool):
    """Sync weights from an FSDP2 trainer into EP=2 inference engines and assert
    the responses are unchanged across the sync.

    grouped_ep=False (the original test): plain-FSDP2 stock-HF trainer (replay/EP off).
    grouped_ep=True (Stage 4b / G4-4): grouped-swapped (``moe_grouped_gemm=True``) +
    EP-sharded (``expert_model_parallel_size=2``) trainer. The EP-sharded grouped
    expert DTensors must reshard (``full_tensor`` across ep+fsdp) AND name/shape-remap
    (``convert_tt_to_hf_moe`` after the shim-prefix strip) back into the EP inference
    engine's per-expert HF layout.
    """
    _check_gpus(num_gpus=NUM_GPUS)

    pg = None
    try:
        cfg = _get_test_cfg()
        cfg.trainer.placement.colocate_all = True
        if grouped_ep:
            # Grouped-GEMM swap + EP=2 on the trainer side (Stage 4b).
            cfg.trainer.policy.fsdp_config.moe_grouped_gemm = True
            cfg.trainer.policy.fsdp_config.expert_model_parallel_size = 2
        # Deterministic sampling for robust comparisons
        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1.0
        cfg.generator.sampling_params.top_k = -1

        initialize_ray(cfg)

        # Create a shared PG with 2 bundles (sufficient for two engines with tp=2 and training)
        pg = placement_group([{"GPU": 1, "CPU": 1}] * NUM_GPUS, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=60)

        # Spin up two inference engines with EP enabled, colocated
        client = init_ray_inference_engines(
            backend=cfg.generator.backend,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            shared_pg=pg,
            config=cfg,
        )
        asyncio.run(client.wake_up())

        # Generate before weight sync
        prompts = get_test_prompts(MODEL, num_samples=4)
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        out_before = asyncio.run(
            client.generate(InferenceEngineInput(prompts=prompts, sampling_params=sampling_params))
        )
        assert len(out_before["responses"]) == len(prompts)

        asyncio.run(client.sleep())

        # Initialize policy worker
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=True,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Sync weights to inference engines
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.wake_up(tags=["weights"]))
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))
        policy.offload_to_cpu()
        asyncio.run(client.wake_up(tags=["kv_cache"]))
        asyncio.run(client.reset_prefix_cache())

        # Generate after weight sync
        out_after = asyncio.run(client.generate(InferenceEngineInput(prompts=prompts, sampling_params=sampling_params)))
        assert len(out_after["responses"]) == len(prompts)
        assert len(out_after["stop_reasons"]) == len(prompts)

        # Check that weights are not corrupted: responses should be similar pre/post sync
        num_similar = 0
        for i in range(len(prompts)):
            if are_responses_similar([out_before["responses"][i]], [out_after["responses"][i]], tolerance=0.02):
                num_similar += 1
            else:
                print(
                    f"Response changed significantly after weight sync: before={out_before['responses'][i][:200]} ... after={out_after['responses'][i][:200]} ..."
                )
        if grouped_ep:
            # G4-4: the EP-sharded grouped trainer weights must reshard + remap into the
            # EP inference engine and reproduce the pre-sync responses.
            assert num_similar == len(prompts), (
                f"G4-4 weight-sync (grouped+EP) corrupted weights: only {num_similar}/{len(prompts)} "
                "responses matched after sync."
            )
    finally:
        if pg is not None:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()


def test_ep_weight_sync():
    """Plain-FSDP2 stock-HF trainer → EP=2 inference (flag-off extractor path)."""
    _run_ep_weight_sync(grouped_ep=False)


def test_ep_weight_sync_grouped():
    """G4-4: grouped-swapped + EP-sharded FSDP2 trainer → EP=2 inference (Stage 4b)."""
    _run_ep_weight_sync(grouped_ep=True)
