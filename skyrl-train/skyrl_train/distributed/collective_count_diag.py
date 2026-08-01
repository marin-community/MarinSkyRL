"""Low-overhead process-group sequence snapshots at policy phase boundaries."""

from __future__ import annotations

import itertools
import json
import os
import threading
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from loguru import logger

_ENV = "SKYRL_COLLECTIVE_COUNT_DIAG"
_LOG_PREFIX = "COLLECTIVE_PHASE_DIAG "
_region_ids = itertools.count(1)
_region_ids_lock = threading.Lock()


class ProcessGroupLike(Protocol):
    def _get_sequence_number_for_group(self) -> int: ...


class DeviceMeshLike(Protocol):
    mesh_dim_names: tuple[str, ...]
    shape: tuple[int, ...]

    def get_coordinate(self) -> list[int]: ...

    def get_group(self, name: str) -> ProcessGroupLike: ...


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
    sequence_numbers: dict[str, int | None]


@dataclass
class _RegionContext:
    region_id: int
    kind: str
    rank: int
    metadata: dict[str, Any]
    device_mesh: DeviceMeshLike
    moe_boundary_logged: bool = False
    next_event_index: int = 0
    capture_error_reported: bool = False


@dataclass(frozen=True)
class CollectivePhaseDivergence:
    region_id: int
    event_index: int
    phase: str
    group_name: str
    group_coordinate: tuple[tuple[str, int], ...]
    reference_rank: int
    divergent_rank: int
    expected: int | str | None
    actual: int | str | None


_region: ContextVar[_RegionContext | None] = ContextVar("collective_phase_region", default=None)


def enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _default_process_group() -> ProcessGroupLike | None:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None
        return dist.distributed_c10d._get_default_group()
    except Exception:
        return None


def _sequence_number(group: ProcessGroupLike | None) -> int | None:
    if group is None:
        return None
    try:
        return int(group._get_sequence_number_for_group())
    except Exception:
        return None


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


def _capture_record(context: _RegionContext, phase: str) -> CollectivePhaseRecord:
    mesh = context.device_mesh
    dim_names = tuple(mesh.mesh_dim_names)
    sequence_numbers = {"world": _sequence_number(_default_process_group())}
    for name in dim_names:
        try:
            group = mesh.get_group(name)
        except Exception:
            group = None
        sequence_numbers[name] = _sequence_number(group)
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


def _serialize_record(record: CollectivePhaseRecord) -> str:
    return _LOG_PREFIX + json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))


def parse_log_line(line: str) -> CollectivePhaseRecord:
    """Parse a phase record from a raw or loguru-decorated worker log line."""
    marker = line.find(_LOG_PREFIX)
    if marker < 0:
        raise ValueError("line does not contain a collective phase diagnostic")
    payload = json.loads(line[marker + len(_LOG_PREFIX) :])
    return CollectivePhaseRecord(
        region_id=int(payload["region_id"]),
        event_index=int(payload["event_index"]),
        kind=str(payload["kind"]),
        rank=int(payload["rank"]),
        phase=str(payload["phase"]),
        metadata=dict(payload["metadata"]),
        mesh_dim_names=tuple(payload["mesh_dim_names"]),
        mesh_shape=tuple(int(size) for size in payload["mesh_shape"]),
        mesh_coordinate=tuple(int(coordinate) for coordinate in payload["mesh_coordinate"]),
        sequence_numbers={
            name: value if value is None else int(value) for name, value in payload["sequence_numbers"].items()
        },
    )


def _group_coordinate(record: CollectivePhaseRecord, group_name: str) -> tuple[tuple[str, int], ...]:
    if group_name == "world":
        return ()
    return tuple(
        (name, coordinate)
        for name, coordinate in zip(record.mesh_dim_names, record.mesh_coordinate)
        if name != group_name
    )


def find_first_divergence(records: list[CollectivePhaseRecord]) -> CollectivePhaseDivergence | None:
    """Find the earliest phase or sequence mismatch represented by rank records."""
    if not records:
        return None
    topology: dict[int, CollectivePhaseRecord] = {}
    for record in records:
        topology.setdefault(record.rank, record)
    mesh_dim_names = records[0].mesh_dim_names
    mesh_shape = records[0].mesh_shape
    if any(record.mesh_dim_names != mesh_dim_names or record.mesh_shape != mesh_shape for record in records):
        raise ValueError("collective phase records use different mesh definitions")

    records_by_event = {(record.region_id, record.event_index, record.rank): record for record in records}
    event_keys = sorted({(record.region_id, record.event_index) for record in records})
    for region_id, event_index in event_keys:
        for group_name in ("world", *mesh_dim_names):
            groups: dict[tuple[tuple[str, int], ...], list[int]] = {}
            for rank, topology_record in topology.items():
                groups.setdefault(_group_coordinate(topology_record, group_name), []).append(rank)
            for group_coordinate, member_ranks in sorted(groups.items()):
                member_ranks.sort()
                available = [
                    records_by_event[(region_id, event_index, rank)]
                    for rank in member_ranks
                    if (region_id, event_index, rank) in records_by_event
                ]
                if not available:
                    continue
                reference = available[0]
                for rank in member_ranks:
                    candidate = records_by_event.get((region_id, event_index, rank))
                    if candidate is None:
                        return CollectivePhaseDivergence(
                            region_id,
                            event_index,
                            reference.phase,
                            group_name,
                            group_coordinate,
                            reference.rank,
                            rank,
                            reference.sequence_numbers[group_name],
                            "<missing>",
                        )
                    if candidate.phase != reference.phase:
                        return CollectivePhaseDivergence(
                            region_id,
                            event_index,
                            reference.phase,
                            group_name,
                            group_coordinate,
                            reference.rank,
                            rank,
                            reference.phase,
                            candidate.phase,
                        )
                    expected = reference.sequence_numbers[group_name]
                    actual = candidate.sequence_numbers[group_name]
                    if actual != expected:
                        return CollectivePhaseDivergence(
                            region_id,
                            event_index,
                            reference.phase,
                            group_name,
                            group_coordinate,
                            reference.rank,
                            rank,
                            expected,
                            actual,
                        )
    return None


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
        logger.info(_serialize_record(record))
        return record
    except Exception as error:
        if not context.capture_error_reported:
            context.capture_error_reported = True
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
