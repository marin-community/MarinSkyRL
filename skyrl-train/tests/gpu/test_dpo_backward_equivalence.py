"""Distributed DPO backward-pass equivalence against an eager HF reference (issue 969).

Replays one fixed preference batch through the production DPO objective on Megatron
pipeline/expert/data-parallel topologies and compares every exported gradient tensor with
an independent float32 eager HF GrugMoE computation of the same DPO loss. The comparison
covers micro-batch scaling (the loss scale that cancels the scheduler's micro-batch
division and the DP gradient average), gradient reduction across pipeline stages and
expert shards, and the batch-major output projection installed by the training worker.

Alignment: named tensors are aligned through the bridge conversion tasks — the same
mapping the weight sync uses — by temporarily exporting ``param.main_grad`` in place of
parameter data. Expert identifiers resolve through the stacked-expert export mapping;
pipeline partitions through the collective export. The HF reference runs in float32 on
CPU; a same-implementation CPU-vs-GPU log-probability agreement check on rank 0 separates
route-flip artifacts from genuine mismatches.

Scope: the production BFCL run used TP1/PP15/CP1/EP8 at bf16 with the fused attention
kernels. This experiment runs TP1/CP1 (the topology condition of the output-projection
adaptation) at fp32 with the unfused attention backend so gradient agreement is tight;
the production bf16 kernels are covered separately by the forward parity gates in
test_grug_megatron.py. Pipeline and expert degrees here do not reach 15/8, so
production-degree equivalence is not established by this test alone.
"""

from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from transformers import AutoTokenizer

import skyrl_train
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper, MegatronPolicyMicroBatch
from tests.gpu.dpo_equivalence import (
    DPOReference,
    PairBatch,
    build_grug_megatron_chunks,
    dpo_gradient_names,
    dpo_wrapper_config,
    export_parameter_gradients,
    fixed_pair_batch,
    hf_dpo_reference,
)
from tests.gpu.grug_gpu_gates import require_hoppers
from tests.gpu.grug_tiny_checkpoint import write_tiny_checkpoint
from tests.gpu.utils import run_distributed_with_deadline

# Registered before execution; see ci/dpo_equivalence/README.md for the measured values.
GRADIENT_RTOL = 2e-4
GRADIENT_ATOL = 3e-4
METRIC_ATOL = 5e-4
# Same-implementation cross-device agreement bound for the HF reference itself.
REFERENCE_DEVICE_ATOL = 1e-4
MICRO_BATCH_SIZE = 2

TOPOLOGIES = [
    (1, 1, 1),
    (2, 2, 1),
    (4, 2, 1),
    (4, 2, 2),
    (8, 2, 4),
]


def _payload(micro: PairBatch, old_log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> MegatronPolicyMicroBatch:
    """Build the production policy micro-batch for one chosen/rejected pair."""
    device = torch.cuda.current_device()
    position_ids = micro.attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(micro.attention_mask == 0, 0)
    zeros = torch.zeros_like(micro.loss_mask)
    return MegatronPolicyMicroBatch(
        sequences=micro.sequences.to(device),
        attention_mask=micro.attention_mask.to(device),
        position_ids=position_ids.to(device),
        num_actions=micro.num_actions,
        old_action_log_probs=old_log_probs.to(device),
        base_action_log_probs=ref_log_probs.to(device),
        advantages=zeros.to(device),
        loss_mask=micro.loss_mask.to(device),
        rollout_action_logprobs=None,
        response_span_tags=None,
        distillation=None,
        ftpo=None,
        correction_weights=None,
        rollout_routed_experts=None,
        pair_roles=micro.pair_roles.to(device),
    )


def _assert_gradient_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = (actual - expected).abs()
    print(
        f"gradient {name}: max_abs={difference.max().item():.8g} mean_abs={difference.mean().item():.8g}",
        flush=True,
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=GRADIENT_RTOL,
        atol=GRADIENT_ATOL,
        msg=lambda message: (
            f"{name}: max_abs={difference.max().item():.8g}, mean_abs={difference.mean().item():.8g}\n{message}"
        ),
    )


def run_backward_rank(rank, world_size, pp, ep, rendezvous, model_path, batch: PairBatch, reference: DPOReference):
    assert Path(skyrl_train.__file__).resolve().parent == Path(__file__).resolve().parents[2] / "skyrl_train"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=1,
            use_sharp=False,
            context_parallel_size=1,
        )
        bridge, provider, chunks, hf_config = build_grug_megatron_chunks(
            model_path,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            bf16=False,
            attention_backend="unfused",
        )
        cfg = dpo_wrapper_config(batch.sequences.shape[1])
        wrapper = MegatronModelWrapper(
            cfg,
            chunks,
            policy_loss_fn=PolicyLossRegistry.get(cfg.trainer.algorithm.policy_loss_type),
            vocabulary_size=hf_config.vocab_size,
        )
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        micros = batch.dp_shard(dp_rank, dp_size, MICRO_BATCH_SIZE)
        payloads = [
            _payload(
                micro,
                reference.action_log_probs[start : start + micro.sequences.shape[0]],
                reference.ref_action_log_probs[start : start + micro.sequences.shape[0]],
            )
            for start, micro in micros
        ]
        metrics = wrapper.forward_backward_mini_batch(
            payloads, seq_len=batch.sequences.shape[1], micro_batch_size=MICRO_BATCH_SIZE
        )
        exports = export_parameter_gradients(bridge, chunks)

        if rank == 0:
            print(
                f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} "
                f"megatron-core={version('megatron-core')} TP=1 PP={pp} EP={ep} DP={dp_size}",
                flush=True,
            )
            no_gradient_names = set(dpo_gradient_names(hf_config.num_hidden_layers))
            exported = set(exports) - no_gradient_names
            missing = set(reference.gradients) - exported
            extra = exported - set(reference.gradients)
            assert not missing, f"reference gradients without a Megatron export: {sorted(missing)}"
            assert not extra, f"Megatron gradient exports absent from the reference: {sorted(extra)}"
            for name in sorted(reference.gradients):
                _assert_gradient_close(name, exports[name], reference.gradients[name])
            worst = max((exports[name] - reference.gradients[name]).abs().max().item() for name in reference.gradients)
            print(f"gradient worst max_abs across all tensors: {worst:.8g}", flush=True)

        # Metric rows pool across data-parallel ranks like the trainer's step metrics.
        for key in reference.metrics:
            actual = torch.tensor(sum(item[key] for item in metrics) / len(metrics), device="cuda")
            dist.all_reduce(actual, group=mpu.get_data_parallel_group(with_context_parallel=False))
            actual /= dp_size
            if rank == 0:
                expected = reference.metrics[key]
                print(f"metric {key}: actual={actual.item():.8g} expected={expected:.8g}", flush=True)
                assert abs(actual.item() - expected) < METRIC_ATOL, (key, actual.item(), expected)

        if rank == 0:
            on_device = hf_dpo_reference(model_path, batch, device="cuda")
            difference = (on_device.action_log_probs - reference.action_log_probs).abs()
            print(
                f"reference CPU-vs-GPU logprobs: max_abs={difference.max().item():.8g} "
                f"mean_abs={difference.mean().item():.8g}",
                flush=True,
            )
            assert difference.max().item() < REFERENCE_DEVICE_ATOL, (
                "fixture is route-sensitive across devices; regenerate with a different seed"
            )
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize(("world_size", "pp", "ep"), TOPOLOGIES, ids=["pp1", "pp2", "pp2_dp2", "pp2_ep2", "pp2_ep4"])
def test_dpo_distributed_backward_matches_hf_reference(tmp_path, world_size, pp, ep):
    require_hoppers(world_size)
    model_path = tmp_path / "model"
    model_path.mkdir()
    write_tiny_checkpoint(model_path)
    pad_token_id = AutoTokenizer.from_pretrained(model_path).pad_token_id
    batch = fixed_pair_batch(pad_token_id)
    reference = hf_dpo_reference(str(model_path), batch)
    print(f"input batch sha256={batch.sha256()} reference dpo/loss={reference.metrics['dpo/loss']:.8g}", flush=True)
    run_distributed_with_deadline(
        run_backward_rank,
        (world_size, pp, ep, (tmp_path / "rendezvous").as_uri(), str(model_path), batch, reference),
        world_size,
    )
