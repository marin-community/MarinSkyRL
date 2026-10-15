"""Megatron's fused projections computed with vLLM's GEMM shapes."""

import torch
import torch.nn.functional as F


def qkv_projection(
    x: torch.Tensor, fused_weight: torch.Tensor, groups: int, query_width: int, head_dim: int
) -> torch.Tensor:
    """Megatron's fused QKV projection computed as compiled vLLM computes q, k and v: three GEMMs.

    Megatron's fused weight holds, for each of ``groups`` KV groups, the group's ``query_width`` query rows, then
    its key and its value rows (``head_dim`` each). vLLM's q, k and v weights are those rows gathered across the
    groups. The result has the fused projection's layout, so Megatron's split reads q, k and v from it.
    """
    grouped = fused_weight.view(groups, query_width + 2 * head_dim, -1)
    weights = (
        grouped[:, :query_width],
        grouped[:, query_width : query_width + head_dim],
        grouped[:, query_width + head_dim :],
    )
    parts = [F.linear(x, weight.reshape(-1, weight.shape[-1])) for weight in weights]
    return torch.cat([part.view(*x.shape[:-1], groups, -1) for part in parts], dim=-1).view(*x.shape[:-1], -1)


def split_qkv(mixed_qkv: torch.Tensor, groups: int, query_width: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    """Megatron's split of the fused ``[S, B, groups * (query_width + 2 * head_dim)]`` projection into query
    ``[S, B, heads, head_dim]``, key and value ``[S, B, groups, head_dim]``."""
    grouped = mixed_qkv.view(*mixed_qkv.shape[:-1], groups, query_width + 2 * head_dim)
    query, key, value = torch.split(grouped, [query_width, head_dim, head_dim], dim=-1)
    return query.reshape(*mixed_qkv.shape[:-1], -1, head_dim), key, value
