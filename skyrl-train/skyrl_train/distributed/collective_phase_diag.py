"""Low-overhead process-group sequence snapshots at policy phase boundaries."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import torch.distributed as dist
from loguru import logger

_ENV = "SKYRL_COLLECTIVE_COUNT_DIAG"
_LOG_PREFIX = "COLLECTIVE_PHASE_DIAG "
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
    capture_error_reported: bool = False


class DivergenceKind(StrEnum):
    MISSING_RANK = "missing_rank"
    PHASE = "phase"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class CollectivePhaseDivergence:
    region_id: int
    event_index: int
    phase: str
    group_name: str
    group_coordinate: tuple[tuple[str, int], ...]
    reference_rank: int
    divergent_rank: int
    kind: DivergenceKind
    expected_phase: str | None = None
    actual_phase: str | None = None
    expected_sequence: int | None = None
    actual_sequence: int | None = None


_region: ContextVar[_RegionContext | None] = ContextVar("collective_phase_region", default=None)


def enabled() -> bool:
    return os.environ.get(_ENV, "0") == "1"


def _default_process_group() -> ProcessGroupLike:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    return dist.distributed_c10d._get_default_group()


def _sequence_number(group: ProcessGroupLike) -> int:
    return int(group._get_sequence_number_for_group())


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
    sequence_numbers = {WORLD_GROUP: _sequence_number(_default_process_group())}
    for name in dim_names:
        sequence_numbers[name] = _sequence_number(mesh.get_group(name))
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
    return _LOG_PREFIX + json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))


def parse_log_line(line: str) -> CollectivePhaseRecord:
    """Parse a phase record from a raw or loguru-decorated worker log line."""
    marker = line.find(_LOG_PREFIX)
    if marker < 0:
        raise ValueError("line does not contain a collective phase diagnostic")
    payload, _ = json.JSONDecoder().raw_decode(line[marker + len(_LOG_PREFIX) :].lstrip())
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
        sequence_numbers={name: int(value) for name, value in payload["sequence_numbers"].items()},
    )


def _group_coordinate(record: CollectivePhaseRecord, group_name: str) -> tuple[tuple[str, int], ...]:
    if group_name == WORLD_GROUP:
        return ()
    return tuple(
        (name, coordinate)
        for name, coordinate in zip(record.mesh_dim_names, record.mesh_coordinate)
        if name != group_name
    )


def _validate_mesh(records: list[CollectivePhaseRecord]) -> tuple[str, ...]:
    mesh_dim_names = records[0].mesh_dim_names
    mesh_shape = records[0].mesh_shape
    if any(record.mesh_dim_names != mesh_dim_names or record.mesh_shape != mesh_shape for record in records):
        raise ValueError("collective phase records use different mesh definitions")
    return mesh_dim_names


def _partition_ranks(
    topology: dict[int, CollectivePhaseRecord],
    group_names: tuple[str, ...],
) -> dict[str, dict[tuple[tuple[str, int], ...], tuple[int, ...]]]:
    partitions: dict[str, dict[tuple[tuple[str, int], ...], tuple[int, ...]]] = {}
    for group_name in group_names:
        members: dict[tuple[tuple[str, int], ...], list[int]] = {}
        for rank, record in topology.items():
            members.setdefault(_group_coordinate(record, group_name), []).append(rank)
        partitions[group_name] = {
            coordinate: tuple(sorted(ranks)) for coordinate, ranks in sorted(members.items())
        }
    return partitions


def _divergence(
    reference: CollectivePhaseRecord,
    *,
    divergent_rank: int,
    group_name: str,
    group_coordinate: tuple[tuple[str, int], ...],
    kind: DivergenceKind,
    candidate: CollectivePhaseRecord | None = None,
) -> CollectivePhaseDivergence:
    return CollectivePhaseDivergence(
        region_id=reference.region_id,
        event_index=reference.event_index,
        phase=reference.phase,
        group_name=group_name,
        group_coordinate=group_coordinate,
        reference_rank=reference.rank,
        divergent_rank=divergent_rank,
        kind=kind,
        expected_phase=reference.phase if kind != DivergenceKind.SEQUENCE else None,
        actual_phase=candidate.phase if candidate is not None and kind == DivergenceKind.PHASE else None,
        expected_sequence=reference.sequence_numbers[group_name]
        if kind != DivergenceKind.PHASE
        else None,
        actual_sequence=candidate.sequence_numbers[group_name]
        if candidate is not None and kind == DivergenceKind.SEQUENCE
        else None,
    )


def _compare_group_event(
    records_by_rank: dict[int, CollectivePhaseRecord],
    member_ranks: tuple[int, ...],
    group_name: str,
    group_coordinate: tuple[tuple[str, int], ...],
) -> CollectivePhaseDivergence | None:
    available = [records_by_rank[rank] for rank in member_ranks if rank in records_by_rank]
    if not available:
        return None
    reference = available[0]
    for rank in member_ranks:
        candidate = records_by_rank.get(rank)
        if candidate is None:
            return _divergence(
                reference,
                divergent_rank=rank,
                group_name=group_name,
                group_coordinate=group_coordinate,
                kind=DivergenceKind.MISSING_RANK,
            )
        if candidate.phase != reference.phase:
            return _divergence(
                reference,
                divergent_rank=rank,
                group_name=group_name,
                group_coordinate=group_coordinate,
                kind=DivergenceKind.PHASE,
                candidate=candidate,
            )
        if candidate.sequence_numbers[group_name] != reference.sequence_numbers[group_name]:
            return _divergence(
                reference,
                divergent_rank=rank,
                group_name=group_name,
                group_coordinate=group_coordinate,
                kind=DivergenceKind.SEQUENCE,
                candidate=candidate,
            )
    return None


def find_first_divergence(records: list[CollectivePhaseRecord]) -> CollectivePhaseDivergence | None:
    """Find the earliest phase or sequence mismatch represented by rank records."""
    if not records:
        return None
    mesh_dim_names = _validate_mesh(records)
    topology: dict[int, CollectivePhaseRecord] = {}
    for record in records:
        topology.setdefault(record.rank, record)
    partitions = _partition_ranks(topology, (WORLD_GROUP, *mesh_dim_names))

    records_by_event: dict[tuple[int, int], dict[int, CollectivePhaseRecord]] = {}
    for record in records:
        records_by_event.setdefault((record.region_id, record.event_index), {})[record.rank] = record
    for event_key in sorted(records_by_event):
        event_records = records_by_event[event_key]
        for group_name, groups in partitions.items():
            for group_coordinate, member_ranks in groups.items():
                divergence = _compare_group_event(event_records, member_ranks, group_name, group_coordinate)
                if divergence is not None:
                    return divergence
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
        logger.info(format_log_record(record))
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


def parse_log_lines(lines: Iterable[str]) -> list[CollectivePhaseRecord]:
    """Extract collective phase records from mixed worker log lines."""
    return [parse_log_line(line) for line in lines if _LOG_PREFIX in line]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report the first collective phase divergence in worker logs.")
    parser.add_argument("logs", nargs="*", type=Path, help="Worker log files. Reads stdin when omitted.")
    args = parser.parse_args(argv)
    if args.logs:
        records = []
        for path in args.logs:
            with path.open(encoding="utf-8", errors="replace") as lines:
                records.extend(parse_log_lines(lines))
    else:
        records = parse_log_lines(sys.stdin)
    divergence = find_first_divergence(records)
    print(
        json.dumps(
            {"record_count": len(records), "divergence": asdict(divergence) if divergence is not None else None},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
