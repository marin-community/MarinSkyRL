"""Low-overhead process-group sequence snapshots at policy phase boundaries."""

from __future__ import annotations

import itertools
import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch.distributed as dist
from loguru import logger

_ENV = "SKYRL_COLLECTIVE_PHASE_DIAGNOSTICS"
LOG_PREFIX = "COLLECTIVE_PHASE_DIAGNOSTICS "
WORLD_GROUP = "world"
_region_ids = itertools.count(1)
_region_ids_lock = threading.Lock()


class ProcessGroupLike(Protocol):
    def _get_sequence_number_for_group(self) -> int: ...


class DeviceMeshLike(Protocol):
    mesh_dim_names: tuple[str, ...]
    shape: tuple[int, ...]

    def get_coordinate(self) -> list[int]: ...

    def get_group(self, name: str) -> ProcessGroupLike: ...


@runtime_checkable
class DeviceMeshStrategy(Protocol):
    device_mesh: DeviceMeshLike | None


@dataclass(frozen=True)
class CollectivePhaseRecord:
    region_id: int
    event_index: int
    kind: str
    rank: int
    phase: str
    metadata: dict[str, Any]
    mesh_dim_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]
    mesh_coordinate: tuple[int, ...]
    sequence_numbers: dict[str, int]


@dataclass
class _RegionContext:
    region_id: int
    kind: str
    rank: int
    metadata: dict[str, Any]
    device_mesh: DeviceMeshLike
    moe_boundary_logged: bool = False
    next_event_index: int = 0


_region: ContextVar[_RegionContext | None] = ContextVar("collective_phase_region", default=None)
_ReturnT = TypeVar("_ReturnT")


def enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _default_process_group() -> ProcessGroupLike:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    return dist.distributed_c10d._get_default_group()


@contextmanager
def region(
    device_mesh: DeviceMeshLike | None,
    *,
    kind: str,
    rank: int,
    metadata: dict[str, Any] | None = None,
) -> Iterator[int | None]:
    """Scope one policy operation, yielding ``None`` when diagnostics are disabled."""
    if not enabled():
        yield None
        return
    if device_mesh is None:
        raise ValueError("enabled collective phase diagnostics require a device mesh")
    with _region_ids_lock:
        region_id = next(_region_ids)
    token = _region.set(
        _RegionContext(
            region_id=region_id,
            kind=kind,
            rank=rank,
            metadata=dict(metadata or {}),
            device_mesh=device_mesh,
        )
    )
    try:
        yield region_id
    finally:
        _region.reset(token)


def with_policy_training_region(function: Callable[..., _ReturnT]) -> Callable[..., _ReturnT]:
    """Run a policy training-step method inside an optional diagnostic region."""

    @wraps(function)
    def wrapped(worker, experience, global_step, local_step, accumulation_steps):
        strategy = worker.strategy
        if not enabled() or not isinstance(strategy, DeviceMeshStrategy) or strategy.device_mesh is None:
            return function(worker, experience, global_step, local_step, accumulation_steps)
        with region(
            strategy.device_mesh,
            kind="policy_training_step",
            rank=worker._rank,
            metadata={"global_step": global_step, "local_step": local_step},
        ):
            return function(worker, experience, global_step, local_step, accumulation_steps)

    return wrapped


def _capture_record(context: _RegionContext, phase: str) -> CollectivePhaseRecord:
    mesh = context.device_mesh
    dim_names = tuple(mesh.mesh_dim_names)
    sequence_numbers = {WORLD_GROUP: int(_default_process_group()._get_sequence_number_for_group())}
    for name in dim_names:
        sequence_numbers[name] = int(mesh.get_group(name)._get_sequence_number_for_group())
    event_index = context.next_event_index
    context.next_event_index += 1
    return CollectivePhaseRecord(
        region_id=context.region_id,
        event_index=event_index,
        kind=context.kind,
        rank=context.rank,
        phase=phase,
        metadata=dict(context.metadata),
        mesh_dim_names=dim_names,
        mesh_shape=tuple(int(size) for size in mesh.shape),
        mesh_coordinate=tuple(int(coordinate) for coordinate in mesh.get_coordinate()),
        sequence_numbers=sequence_numbers,
    )


def format_log_record(record: CollectivePhaseRecord) -> str:
    return LOG_PREFIX + json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))


def log_phase(phase: str, *, reset_moe_boundary: bool = False) -> CollectivePhaseRecord | None:
    """Log counters, or return ``None`` when disabled, unscoped, or capture fails."""
    if not enabled():
        return None
    context = _region.get()
    if context is None:
        return None
    if reset_moe_boundary:
        context.moe_boundary_logged = False
    try:
        record = _capture_record(context, phase)
        logger.info(format_log_record(record))
        return record
    except Exception as error:
        # This opt-in diagnostic must not replace the original training failure.
        # Every failed boundary remains visible rather than becoming a null counter.
        logger.warning(f"Collective phase diagnostic failed in region={context.region_id} phase={phase}: {error!r}")
        return None


def log_moe_ep_boundary_once() -> CollectivePhaseRecord | None:
    """Log the first MoE boundary, or return ``None`` when inactive or already logged."""
    if not enabled():
        return None
    context = _region.get()
    if context is None or context.moe_boundary_logged:
        return None
    context.moe_boundary_logged = True
    return log_phase("moe_ep_a2a_first")
