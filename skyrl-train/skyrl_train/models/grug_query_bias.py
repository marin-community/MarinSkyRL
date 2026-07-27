# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""Query-bias observations and optimizer-window reduction for Grug MoE."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def query_bias_candidate_count(valid_tokens: int, experts_per_token: int, num_experts: int) -> int:
    """Return Grug's per-expert candidate count for one optimizer window."""

    if valid_tokens < 1:
        raise ValueError(f"valid_tokens must be positive, got {valid_tokens}")
    return max(1, valid_tokens * experts_per_token // num_experts)


@dataclass(frozen=True)
class GrugQueryBiasLayerObservation:
    """Compact routing data from one layer and one microbatch.

    ``candidates`` contains the largest ``q`` values of ``s - alpha`` for each
    expert. Keeping only these values is sufficient to recover the exact q-th
    value across a gradient-accumulation window.
    """

    candidates: torch.Tensor
    selected_experts: torch.Tensor
    combine_weights: torch.Tensor


@dataclass(frozen=True)
class GrugQueryBiasObservation:
    layers: tuple[GrugQueryBiasLayerObservation, ...]
    candidate_count: int


class GrugQueryBiasAccumulator:
    """Retain exact per-expert top-q candidates across one optimizer window."""

    def __init__(self, *, candidate_count: int, num_layers: int, num_experts: int) -> None:
        if candidate_count < 1:
            raise ValueError(f"candidate_count must be positive, got {candidate_count}")
        self.candidate_count = candidate_count
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._candidates: list[torch.Tensor | None] = [None] * num_layers

    def observe(self, observation: GrugQueryBiasObservation) -> None:
        if observation.candidate_count != self.candidate_count:
            raise ValueError(
                f"observation q={observation.candidate_count} does not match window q={self.candidate_count}"
            )
        if len(observation.layers) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer observations, got {len(observation.layers)}")

        for layer_idx, layer in enumerate(observation.layers):
            candidates = layer.candidates.float()
            if candidates.ndim != 2 or candidates.shape[0] != self.num_experts:
                raise ValueError(
                    f"layer {layer_idx} candidates must have shape [{self.num_experts}, q], "
                    f"got {tuple(candidates.shape)}"
                )
            previous = self._candidates[layer_idx]
            combined = candidates if previous is None else torch.cat((previous, candidates), dim=-1)
            keep = min(self.candidate_count, combined.shape[-1])
            self._candidates[layer_idx] = torch.topk(combined, k=keep, dim=-1, sorted=True).values

    @torch.no_grad()
    def finalize_betas(self) -> torch.Tensor:
        """Return ``[layers, experts]`` local q-th values, averaged across ranks."""

        if any(candidates is None for candidates in self._candidates):
            raise RuntimeError("cannot finalize query bias before observing every layer")

        betas = []
        for layer_idx, candidates in enumerate(self._candidates):
            assert candidates is not None
            if candidates.shape[-1] < self.candidate_count:
                raise RuntimeError(
                    f"layer {layer_idx} saw only {candidates.shape[-1]} candidates for q={self.candidate_count}"
                )
            betas.append(candidates[:, self.candidate_count - 1])
        result = torch.stack(betas, dim=0).float()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
            result.div_(world_size)
        return result


def next_query_bias(betas: torch.Tensor) -> torch.Tensor:
    """Apply Grug's ``bias = -beta`` update and center each layer."""

    if betas.ndim != 2:
        raise ValueError(f"betas must have shape [layers, experts], got {tuple(betas.shape)}")
    bias = -betas.float()
    return bias - bias.mean(dim=-1, keepdim=True)
