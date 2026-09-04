"""Shared token routing for grouped mixture-of-experts implementations."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from torch import nn


class GroupedExpertCallable(Protocol):
    """Expert holder accepted by the grouped routing pipeline."""

    def __call__(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor: ...


class GroupedRouting(NamedTuple):
    """Named outputs from grouping token routes by expert."""

    scores: torch.Tensor
    token_indices: torch.Tensor
    tokens_per_expert: torch.Tensor


class TokenReorderer(nn.Module):
    """Reorder token indices to match expert ordering for grouped compute."""

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> GroupedRouting:
        selected_experts_indices = selected_experts_indices.reshape(-1)
        # Integer counts are required by both torch.split and EP all-to-all.
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        ).to(torch.int64)
        sorted_route_indices = torch.argsort(selected_experts_indices, stable=True)
        sorted_scores = top_scores.reshape(-1)[sorted_route_indices]
        sorted_token_indices = sorted_route_indices // self.top_k
        return GroupedRouting(sorted_scores, sorted_token_indices, num_tokens_per_expert)


def run_experts_for_loop(
    gate_weights: torch.Tensor,
    down_weights: torch.Tensor,
    up_weights: torch.Tensor,
    routed_input: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Match eager gated-MLP numerics and zero any padded routed rows."""

    counts = num_tokens_per_expert.to(torch.int64).tolist()
    routed_rows = sum(counts)
    num_padding = routed_input.shape[0] - routed_rows
    input_splits = torch.split(routed_input[:routed_rows], split_size_or_sections=counts, dim=0)
    output_splits = []
    for expert_idx, expert_input in enumerate(input_splits):
        gate = F.silu(torch.matmul(expert_input, gate_weights[expert_idx].transpose(-2, -1)))
        up = torch.matmul(expert_input, up_weights[expert_idx].transpose(-2, -1))
        hidden = gate * up
        output_splits.append(torch.matmul(hidden, down_weights[expert_idx].transpose(-2, -1)))
    output = torch.cat(output_splits, dim=0)
    return torch.vstack((output, output.new_zeros((num_padding, output.shape[-1]))))


def grouped_expert_contributions(
    experts: GroupedExpertCallable,
    hidden_states: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    reorderer: TokenReorderer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each routed row's token index and its weighted expert output, grouped by expert."""

    routing = reorderer(
        top_scores,
        selected_experts_indices,
    )
    hidden_size = hidden_states.shape[-1]
    routed_input = torch.gather(
        hidden_states, dim=0, index=routing.token_indices.reshape(-1, 1).expand(-1, hidden_size)
    )
    routed_output = experts(routed_input, routing.tokens_per_expert)
    routed_output = (routed_output.float() * routing.scores.reshape(-1, 1)).to(hidden_states.dtype)
    return routing.token_indices, routed_output


def combine_routed_rows(
    routed_output: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Sum each token's ``top_k`` routed rows in float32, in ascending expert order.

    A ``scatter_add`` over the repeated token indices reduces with atomics whose order changes from
    launch to launch, and a float32 sum of bf16 rows is inexact once their exponents span more than
    14, so the old-log-prob forward and the training forward could disagree in the last bit. The rows
    arrive grouped by expert, so a stable sort by token leaves every token's rows in expert order and
    the chain of adds below reduces them the same way on every device and every shape.
    """

    order = torch.argsort(token_indices, stable=True)
    per_token = routed_output.index_select(0, order).view(num_tokens, top_k, routed_output.shape[-1])
    combined = per_token[:, 0].float()
    for slot in range(1, top_k):
        combined = combined + per_token[:, slot]
    return combined
