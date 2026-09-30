"""Emulate vLLM's expert-parallel MoE combine for one token set on one GPU.

With DP and EP over ``R`` ranks and the all-gather/reduce-scatter transport, every rank runs its local
experts on all gathered tokens, sums its local top-k slots per token in fp32 and rounds once to bf16,
and a bf16 reduce-scatter then adds the ``R`` per-rank partials and returns each token's sum to the
rank that holds its request. Each addition of the reduce-scatter rounds to bf16, so the result depends
on the order in which the partials meet.
"""

from __future__ import annotations

from enum import StrEnum

import torch


class ReduceOrder(StrEnum):
    RING = "ring"
    """NCCL ring reduce-scatter: the home rank's chunk starts at the next rank and reaches home last."""
    RANK = "rank"
    """Partials added in rank order 0, 1, ..., R-1."""


def expert_map(num_experts: int, ep_size: int, ep_rank: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """vLLM's linear expert placement: global expert id to this rank's local index, or -1 when remote."""
    if num_experts % ep_size:
        raise ValueError(f"{num_experts} experts do not divide over {ep_size} ranks")
    per_rank = num_experts // ep_size
    mapping = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    mapping[ep_rank * per_rank : (ep_rank + 1) * per_rank] = torch.arange(per_rank, dtype=torch.int32, device=device)
    return mapping


def reduction_order(order: ReduceOrder, ep_size: int, home_rank: int) -> list[int]:
    if order is ReduceOrder.RING:
        return [(home_rank + 1 + step) % ep_size for step in range(ep_size)]
    return list(range(ep_size))


def reduce_partials(partials: torch.Tensor, order: list[int]) -> torch.Tensor:
    """Add ``partials[order[0]] + partials[order[1]] + ...`` left to right, rounding each sum to the partials' dtype.

    ``partials`` is ``[ranks, tokens, hidden]``. Each step adds two bf16 values in fp32 and rounds once,
    which equals the correctly rounded bf16 sum that NCCL's bf16 reduction produces.
    """
    if sorted(order) != list(range(partials.shape[0])):
        raise ValueError(f"order {order} is not a permutation of {partials.shape[0]} ranks")
    total = partials[order[0]]
    for rank in order[1:]:
        total = (total.float() + partials[rank].float()).to(partials.dtype)
    return total
