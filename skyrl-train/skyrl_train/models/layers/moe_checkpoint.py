"""Checkpoint-local replay of MoE routing decisions.

Non-reentrant activation checkpointing runs a decoder layer once in the forward
and again during backward. The router must select the same experts both times:
expert-parallel dispatch derives collective split sizes from those indices.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import nn


@dataclass
class _CheckpointRoutes:
    routes: dict[int, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    replay_positions: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, module: nn.Module, routes: torch.Tensor, num_experts: int) -> None:
        if num_experts > 32768:
            raise ValueError(f"checkpoint route storage supports at most 32768 experts, got {num_experts}")
        # Qwen3-Next has 512 experts, so uint8 is not sufficient for every
        # supported model. Keep the common <=256 case half the size.
        storage_dtype = torch.uint8 if num_experts <= 256 else torch.int16
        self.routes[id(module)].append(routes.detach().to(dtype=storage_dtype))

    def replay(self, module: nn.Module) -> torch.Tensor:
        module_id = id(module)
        position = self.replay_positions[module_id]
        recorded = self.routes.get(module_id, [])
        if position >= len(recorded):
            raise RuntimeError(
                "activation-checkpoint MoE recomputation has no recorded route "
                f"for {type(module).__name__} call {position}"
            )
        self.replay_positions[module_id] += 1
        return recorded[position].to(dtype=torch.long)


_ACTIVE_ROUTES: ContextVar[tuple[_CheckpointRoutes, bool] | None] = ContextVar(
    "moe_activation_checkpoint_routes", default=None
)


@contextmanager
def _route_context(state: _CheckpointRoutes, *, recomputing: bool) -> Iterator[None]:
    if recomputing:
        # A retained graph may be recomputed by more than one backward call.
        state.replay_positions.clear()
    token = _ACTIVE_ROUTES.set((state, recomputing))
    try:
        yield
    finally:
        _ACTIVE_ROUTES.reset(token)


def moe_recompute_context_fn():
    """Return paired checkpoint contexts sharing one forward's expert routes."""
    state = _CheckpointRoutes()
    return _route_context(state, recomputing=False), _route_context(state, recomputing=True)


def get_recomputed_routes(module: nn.Module) -> torch.Tensor | None:
    active = _ACTIVE_ROUTES.get()
    if active is None:
        return None
    state, recomputing = active
    return state.replay(module) if recomputing else None


def record_forward_routes(module: nn.Module, routes: torch.Tensor, num_experts: int) -> None:
    active = _ACTIVE_ROUTES.get()
    if active is None:
        return
    state, recomputing = active
    if not recomputing:
        state.record(module, routes, num_experts)
