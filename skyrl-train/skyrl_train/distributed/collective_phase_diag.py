"""Low-overhead process-group sequence snapshots at policy phase boundaries."""

from __future__ import annotations

import itertools
import json
import os
import threading
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

import torch.distributed as dist
from loguru import logger

_ENV = "SKYRL_COLLECTIVE_PHASE_DIAG"
LOG_PREFIX = "COLLECTIVE_PHASE_DIAG "
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


def enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _default_process_group() -> ProcessGroupLike:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    return dist.distributed_c10d._get_default_group()


def begin_region(
    device_mesh: DeviceMeshLike,
    *,
    kind: str,
    rank: int,
    metadata: dict[str, Any] | None = None,
) -> int | None:
    """Start one policy operation region and return its process-local identifier."""
    if not enabled():
        return None
    with _region_ids_lock:
        region_id = next(_region_ids)
    _region.set(
        _RegionContext(
            region_id=region_id,
            kind=kind,
            rank=rank,
            metadata=dict(metadata or {}),
            device_mesh=device_mesh,
        )
    )
    return region_id


def end_region() -> None:
    """Clear the active diagnostic region in the current execution context."""
    _region.set(None)


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
    """Log the active region's process-group counters without issuing a collective."""
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
    """Log the first MoE boundary after the most recent phase reset."""
    if not enabled():
        return None
    context = _region.get()
    if context is None or context.moe_boundary_logged:
        return None
    context.moe_boundary_logged = True
    return log_phase("moe_ep_a2a_first")
