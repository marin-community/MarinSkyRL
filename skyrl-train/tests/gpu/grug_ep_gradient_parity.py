"""Grug EP expert-gradient parity against an independent eager oracle.

This is a direct two-Hopper regression, not a pytest test. It fixes one logical
batch, scalar objective, routing, and BF16 weights, then reconstructs the full
EP2/FSDP1 ``gate_proj.weight`` gradient without a production gather helper.

Run from ``skyrl-train``::

    torchrun --standalone --nproc-per-node=2 tests/gpu/grug_ep_gradient_parity.py
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.distributed as dist
from skyrl_train.distributed.fsdp_utils import apply_ep, create_device_mesh
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeExperts
from skyrl_train.models.layers.moe_routing import TokenReorderer, grouped_expert_contributions
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from tests.distributed_runtime_constants import GPU_TEST_PROCESS_GROUP_TIMEOUT_SECONDS

WORLD_SIZE = 2
NUM_EXPERTS = 2
TOP_K = 1
TOKENS = 4
HIDDEN = 32
INTERMEDIATE = 16
DENOMINATOR = TOKENS * HIDDEN
# Grouped and eager BF16 kernels may differ in low-order bits; absolute slack
# protects near-zero entries. An unaveraged EP2 gradient is a factor-of-two
# error, well outside both bounds.
ATOL = 1e-4
RTOL = 8e-2


class _Holder(nn.Module):
    def __init__(self, experts: GrugMoeExperts) -> None:
        super().__init__()
        self.experts = experts


def _config() -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        shared_expert_intermediate_size=HIDDEN,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=32,
    )


def _fixed_weights() -> dict[str, torch.Tensor]:
    shapes = {
        "gate_proj.weight": (NUM_EXPERTS, INTERMEDIATE, HIDDEN),
        "up_proj.weight": (NUM_EXPERTS, INTERMEDIATE, HIDDEN),
        "down_proj.weight": (NUM_EXPERTS, HIDDEN, INTERMEDIATE),
    }
    weights = {}
    for index, (name, shape) in enumerate(shapes.items()):
        values = torch.arange(math.prod(shape), dtype=torch.float32)
        weights[name] = (torch.sin(values * 0.017 + index * 0.31) * 0.05).reshape(shape).to(torch.bfloat16)
    return weights


def _fixed_replay(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(TOKENS * HIDDEN, dtype=torch.float32)
    hidden = (torch.sin(values * 0.071) + 0.25 * torch.cos(values * 0.037)).reshape(TOKENS, HIDDEN)
    token_ids = torch.arange(TOKENS, dtype=torch.int64)
    selected = (token_ids % NUM_EXPERTS).unsqueeze(-1)
    scores = (0.75 + 0.01 * token_ids).unsqueeze(-1)
    cotangent = torch.cos(values * 0.043 + 0.2).reshape(TOKENS, HIDDEN)
    return (
        hidden.to(device=device, dtype=torch.bfloat16),
        selected.to(device),
        scores.to(device=device, dtype=torch.float32),
        cotangent.to(device=device, dtype=torch.float32),
    )


def _load_weights(module: GrugMoeExperts, weights: dict[str, torch.Tensor], device: torch.device) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(weights[name].to(device=device))


def _grouped_forward(
    experts: GrugMoeExperts,
    hidden: torch.Tensor,
    selected: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    reorderer = TokenReorderer(NUM_EXPERTS, TOP_K)
    routed_indices, routed_output = grouped_expert_contributions(experts, hidden, scores, selected, reorderer)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    output.scatter_add_(0, routed_indices, routed_output.float())
    return output.to(hidden.dtype)


def _objective(output: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
    return (output.float() * cotangent).sum() / DENOMINATOR


def _owned_row(parameter: torch.Tensor, expected: torch.Tensor) -> int:
    local = parameter.to_local().detach().float()
    if local.shape[0] != 1:
        raise AssertionError(f"expected one local expert row, got {tuple(local.shape)}")
    expected = expected.to(local.device)
    matches = [row for row in range(NUM_EXPERTS) if torch.equal(local[0], expected[row].float())]
    if len(matches) != 1:
        raise AssertionError(f"expected one frozen-weight owner, got {matches}")
    return matches[0]


def _reconstruct_gradient(parameter: torch.Tensor, owned_row: int) -> torch.Tensor:
    local = parameter.grad.to_local().detach().float().contiguous()
    gathered = [torch.empty_like(local) for _ in range(WORLD_SIZE)]
    row = torch.tensor([owned_row], dtype=torch.int64, device=local.device)
    gathered_rows = [torch.empty_like(row) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, local)
    dist.all_gather(gathered_rows, row)
    result = torch.empty((NUM_EXPERTS, *local.shape[1:]), dtype=torch.float32, device=local.device)
    coverage = torch.zeros(NUM_EXPERTS, dtype=torch.int64, device=local.device)
    for shard, rows in zip(gathered, gathered_rows, strict=True):
        result[rows.item()].copy_(shard[0])
        coverage[rows.item()] += 1
    if not torch.equal(coverage, torch.ones_like(coverage)):
        raise AssertionError(f"expert-row coverage is not one-to-one: {coverage.tolist()}")
    return result


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != WORLD_SIZE:
        raise AssertionError(f"expected {WORLD_SIZE} ranks, got {dist.get_world_size()}")

    weights = _fixed_weights()
    hidden, selected, scores, cotangent = _fixed_replay(device)

    reference = GrugMoeExperts(_config()).to(device=device, dtype=torch.bfloat16)
    _load_weights(reference, weights, device)
    reference_loss = _objective(reference.forward_eager(hidden, selected, scores), cotangent)
    reference_loss.backward()
    expected = reference.gate_proj.weight.grad.detach().float()

    distributed = GrugMoeExperts(_config()).to(device=device, dtype=torch.bfloat16)
    _load_weights(distributed, weights, device)
    distributed.use_grouped_mm = True
    mesh = create_device_mesh(
        WORLD_SIZE,
        fsdp_size=1,
        timeout_seconds=GPU_TEST_PROCESS_GROUP_TIMEOUT_SECONDS,
        ep_size=2,
        device_type="cuda",
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )
    sharded = apply_ep(
        _Holder(distributed),
        mesh,
        ep_comm_backend="torch",
        fsdp_kwargs={
            "mesh": mesh["fsdp"],
            "mp_policy": mp_policy,
            "offload_policy": None,
            "reshard_after_forward": True,
        },
    )
    if sharded != 1:
        raise AssertionError(f"expected one sharded expert module, got {sharded}")

    owned_row = _owned_row(distributed.gate_proj.weight, weights["gate_proj.weight"])
    actual_loss = _objective(_grouped_forward(distributed, hidden, selected, scores), cotangent)
    actual_loss.backward()
    actual = _reconstruct_gradient(distributed.gate_proj.weight, owned_row)

    difference = (actual - expected).abs()
    close = torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)
    result = {
        "schema": "grug-ep-gradient-parity-v1",
        "topology": {"world_size": 2, "ep_size": 2, "fsdp_size": 1},
        "dtype": "bfloat16",
        "parameter": "gate_proj.weight",
        "objective_abs_error": float((actual_loss - reference_loss).detach().abs()),
        "gradient": {
            "close": bool(close),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "actual_norm": float(torch.linalg.vector_norm(actual)),
            "expected_norm": float(torch.linalg.vector_norm(expected)),
        },
        "tolerance": {"atol": ATOL, "rtol": RTOL},
    }
    if dist.get_rank() == 0:
        print("GRUG_EP_GRADIENT_PARITY_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not close:
        raise AssertionError(
            f"EP2 logical gradient differs from eager oracle: max={result['gradient']['max_abs']:.6g}, "
            f"mean={result['gradient']['mean_abs']:.6g}"
        )


if __name__ == "__main__":
    main()
