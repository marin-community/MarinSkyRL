"""The per-forward object the exact model owns: what the forward's layers hand each other and keep for the
activation recompute."""

from dataclasses import dataclass, field

import torch

from skyrl_train.trajectory_runners.types import UNKNOWN_ENGINE_DP_RANK


@dataclass(frozen=True)
class ResidualSum:
    """The bf16 tensors a layer added in fp32 to form its output, ``residual + mlp``, detached."""

    residual: torch.Tensor
    mlp: torch.Tensor


@dataclass
class ExactForward:
    """One model forward's hand-offs: each sequence's serving engine rank and the engine's expert-parallel size,
    which order the expert-parallel combine, and the attention outputs a checkpointed layer's first forward keeps
    for its recompute (``keep_attention``), keyed by layer number."""

    serving_ranks: torch.Tensor
    expert_parallel_size: int
    keep_attention: bool
    attention: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.serving_ranks.ndim != 1 or self.serving_ranks.dtype != torch.long:
            raise ValueError(f"serving ranks must be one long per sequence, got {self.serving_ranks}")

    def row_serving_ranks(self, rows: int) -> torch.Tensor:
        """Each of ``rows`` router rows' serving rank: row ``s * B + b`` holds position ``s`` of sequence ``b``."""
        sequences = self.serving_ranks.numel()
        if rows % sequences:
            raise ValueError(f"{rows} router rows do not cover {sequences} sequences evenly")
        return self.serving_ranks.repeat(rows // sequences)

    def kept_attention(self, layer_number: int) -> torch.Tensor | None:
        """The attention output kept for this layer's recompute, taken once; ``None`` in a first forward."""
        return self.attention.pop(layer_number, None)

    def keep(self, layer_number: int, attention: torch.Tensor) -> None:
        if self.keep_attention:
            self.attention[layer_number] = attention

    def assert_drained(self) -> None:
        """Raise unless every attention output kept by a first forward was taken by a recompute."""
        if self.attention:
            raise RuntimeError(f"layers {sorted(self.attention)} kept attention outputs no recompute took")


def check_serving_ranks(ranks: torch.Tensor, scored: torch.Tensor, data_parallel_size: int) -> None:
    """Raise unless every sequence with loss tokens (``scored``, ``[B]`` bool) has a serving engine rank below
    ``data_parallel_size``; a sequence without loss tokens may carry ``UNKNOWN_ENGINE_DP_RANK``."""
    if ranks.shape != scored.shape:
        raise ValueError(f"{ranks.numel()} serving ranks for {scored.numel()} sequences")
    if bool(((ranks < 0) & scored).any()):
        raise ValueError(f"every sequence with loss tokens needs its serving engine rank, got {ranks.tolist()}")
    if bool((ranks < UNKNOWN_ENGINE_DP_RANK).any()) or bool((ranks >= data_parallel_size).any()):
        raise ValueError(
            f"serving engine ranks must be below the engines' data-parallel size {data_parallel_size} or "
            f"{UNKNOWN_ENGINE_DP_RANK}, got {ranks.tolist()}"
        )
