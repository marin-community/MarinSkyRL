"""FSDP2 expert-parallel KL-reference regression.

This opt-in test builds a tiny local Qwen3 MoE checkpoint and sends it through the real reference-worker
initialization and log-probability forward. It requires four GPUs for an EP=2 x FSDP=2 mesh and TorchTitan.

Run from ``skyrl-train/`` in the production FSDP runtime::

    python -m pytest -s tests/gpu/test_fsdp2_ep_kl_reference.py

The test is intentionally outside ``tests/gpu/gpu_ci`` because ordinary PR CI has no multi-GPU allocation.
"""

from __future__ import annotations

import math

import ray
import torch
from ray.util.placement_group import placement_group
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import get_ray_pg_ready_with_timeout, initialize_ray
from tests.gpu.utils import get_available_gpus, get_test_actor_config, init_worker_with_type


NUM_GPUS = 4


def _tiny_qwen3_moe_checkpoint(path) -> None:
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
    )
    config._attn_implementation = "eager"
    Qwen3MoeForCausalLM._from_config(config).save_pretrained(path)


def _reference_forward_batch() -> TrainingInputBatch:
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((2, 6), dtype=torch.long),
        }
    )
    batch.metadata = {"response_length": 3}
    return batch


def test_fsdp2_ep_kl_reference_initializes_and_computes_logprobs(tmp_path):
    available = get_available_gpus()
    if len(available) < NUM_GPUS:
        import pytest

        pytest.skip(f"requires {NUM_GPUS} GPUs, found {len(available)}")

    checkpoint = tmp_path / "tiny-qwen3-moe"
    _tiny_qwen3_moe_checkpoint(checkpoint)

    cfg = get_test_actor_config()
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.flash_attn = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.policy.model.path = str(checkpoint)
    cfg.trainer.ref.model.path = str(checkpoint)
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = NUM_GPUS
    cfg.trainer.ref.fsdp_config.fsdp_size = 2
    cfg.trainer.ref.fsdp_config.expert_model_parallel_size = 2
    cfg.trainer.ref.fsdp_config.moe_grouped_gemm = True
    cfg.trainer.ref.fsdp_config.use_grouped_mm = False
    cfg.trainer.ref.fsdp_config.cpu_offload = False

    pg = None
    try:
        initialize_ray(cfg)
        pg = placement_group([{"GPU": 1, "CPU": 1}] * NUM_GPUS, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=60)
        reference = init_worker_with_type(
            "ref",
            shared_pg=pg,
            colocate_all=True,
            num_gpus_per_node=NUM_GPUS,
            cfg=cfg,
        )

        per_rank = ray.get(reference.async_run_ray_method("mesh", "forward", data=_reference_forward_batch()))
        output = concatenate_outputs_after_mesh_dispatch(reference.actor_infos, per_rank)["output"]

        assert output.shape == (2, 3)
        assert all(math.isfinite(value) for value in output.flatten().tolist())
    finally:
        if pg is not None:
            ray.util.remove_placement_group(pg)
        ray.shutdown()
