"""Emulate vLLM's expert-parallel MoE combine for one token set on one GPU.

With DP and EP over ``R`` ranks and the all-gather/reduce-scatter transport, every rank runs its local
experts on all gathered tokens, sums its local top-k slots per token in fp32 and rounds once to bf16,
and a bf16 reduce-scatter then adds the ``R`` per-rank partials and returns each token's sum to the
rank that holds its request. How the reduce-scatter adds the partials decides the result: a ring
rounds each addition to bf16, so the order in which the partials meet matters; a reduction that
accumulates in fp32 (NCCL's NVLS reduction on NVSwitch systems accumulates bf16 in fp32) rounds once.
With two ranks every model gives the same bytes.
"""

from __future__ import annotations

from enum import StrEnum

import torch


class ReduceOrder(StrEnum):
    RING = "ring"
    """NCCL ring reduce-scatter: the home rank's chunk starts at the next rank and reaches home last."""
    RANK = "rank"
    """Partials added in rank order 0, 1, ..., R-1, each sum rounded to bf16."""
    FP32 = "fp32"
    """All partials summed in fp32 and rounded to bf16 once."""


def reduction_order(order: ReduceOrder, ep_size: int, home_rank: int) -> list[int]:
    """The ranks in the order their partials are added."""
    if order is ReduceOrder.RING:
        return [(home_rank + 1 + step) % ep_size for step in range(ep_size)]
    return list(range(ep_size))


def reduce_partials(partials: torch.Tensor, order: ReduceOrder, home_rank: int) -> torch.Tensor:
    """Sum ``partials`` ``[ranks, tokens, hidden]`` as the reduce-scatter model ``order`` does.

    A ring or rank-order sum adds two values in fp32 and rounds each sum to the partials' dtype, which
    equals the correctly rounded sum a bf16 NCCL reduction step produces.
    """
    ranks = reduction_order(order, partials.shape[0], home_rank)
    if order is ReduceOrder.FP32:
        return partials[ranks].float().sum(dim=0).to(partials.dtype)
    total = partials[ranks[0]]
    for rank in ranks[1:]:
        total = (total.float() + partials[rank].float()).to(partials.dtype)
    return total
