"""Shared assertions and NCCL recording for distributed schedule tests."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from torch.distributed import ProcessGroup


@dataclass(frozen=True)
class CollectiveEvent:
    operation: str
    sequence_number: int


@dataclass(frozen=True)
class CollectiveBoundary:
    label: str
    sequence_numbers: Mapping[str, int]


@dataclass(frozen=True)
class RankCollectiveSchedule:
    rank: int
    mesh_dim_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]
    mesh_coordinate: tuple[int, ...]
    events: Mapping[str, tuple[CollectiveEvent, ...]]
    boundaries: tuple[CollectiveBoundary, ...]


@dataclass(frozen=True)
class CollectiveScheduleDivergence:
    group_dimension: str
    fixed_coordinate: tuple[tuple[str, int], ...]
    reference_rank: int
    divergent_rank: int
    sequence_kind: str
    sequence_index: int
    expected: str
    actual: str


def _group_key(schedule: RankCollectiveSchedule, group_dimension: str) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, coordinate)
        for name, coordinate in zip(schedule.mesh_dim_names, schedule.mesh_coordinate)
        if name != group_dimension
    )


def _normalized_events(schedule: RankCollectiveSchedule, group_dimension: str) -> tuple[str, ...]:
    return tuple(event.operation for event in sorted(schedule.events[group_dimension], key=lambda event: event.sequence_number))


def _normalized_boundaries(
    schedule: RankCollectiveSchedule,
    group_dimension: str,
) -> tuple[str, ...]:
    if not schedule.boundaries:
        return ()
    initial = schedule.boundaries[0].sequence_numbers[group_dimension]
    return tuple(
        f"{boundary.label} at sequence +{boundary.sequence_numbers[group_dimension] - initial}"
        for boundary in schedule.boundaries
    )


def _first_difference(reference: Sequence[str], candidate: Sequence[str]) -> tuple[int, str, str] | None:
    for index in range(max(len(reference), len(candidate))):
        expected = reference[index] if index < len(reference) else "<end>"
        actual = candidate[index] if index < len(candidate) else "<end>"
        if expected != actual:
            return index, expected, actual
    return None


def find_first_collective_divergence(
    schedules: Sequence[RankCollectiveSchedule],
    group_dimension: str,
) -> CollectiveScheduleDivergence | None:
    """Return the first operation or boundary mismatch within any mesh process group."""

    if not schedules:
        raise ValueError("at least one rank schedule is required")
    mesh_dim_names = schedules[0].mesh_dim_names
    if group_dimension not in mesh_dim_names:
        raise ValueError(f"unknown mesh dimension {group_dimension!r}; expected one of {mesh_dim_names}")
    dimension_index = mesh_dim_names.index(group_dimension)

    grouped: dict[tuple[tuple[str, int], ...], list[RankCollectiveSchedule]] = {}
    for schedule in schedules:
        if schedule.mesh_dim_names != mesh_dim_names or schedule.mesh_shape != schedules[0].mesh_shape:
            raise ValueError("rank schedules use different mesh definitions")
        grouped.setdefault(_group_key(schedule, group_dimension), []).append(schedule)

    expected_group_size = schedules[0].mesh_shape[dimension_index]
    for fixed_coordinate, members in sorted(grouped.items()):
        members.sort(key=lambda schedule: schedule.rank)
        if len(members) != expected_group_size:
            raise ValueError(
                f"{group_dimension} group {fixed_coordinate} has {len(members)} schedules; "
                f"expected {expected_group_size}"
            )
        reference = members[0]
        reference_operations = _normalized_events(reference, group_dimension)
        reference_boundaries = _normalized_boundaries(reference, group_dimension)
        for candidate in members[1:]:
            comparisons = (
                ("operation", reference_operations, _normalized_events(candidate, group_dimension)),
                ("boundary", reference_boundaries, _normalized_boundaries(candidate, group_dimension)),
            )
            for sequence_kind, expected_sequence, actual_sequence in comparisons:
                difference = _first_difference(expected_sequence, actual_sequence)
                if difference is None:
                    continue
                index, expected, actual = difference
                return CollectiveScheduleDivergence(
                    group_dimension,
                    fixed_coordinate,
                    reference.rank,
                    candidate.rank,
                    sequence_kind,
                    index,
                    expected,
                    actual,
                )
    return None


def assert_collective_schedules_match(
    schedules: Sequence[RankCollectiveSchedule],
    group_dimension: str,
) -> None:
    divergence = find_first_collective_divergence(schedules, group_dimension)
    if divergence is None:
        return
    raise AssertionError(
        f"{divergence.group_dimension} collective schedule diverged for "
        f"{dict(divergence.fixed_coordinate)} at {divergence.sequence_kind} {divergence.sequence_index}: "
        f"rank {divergence.reference_rank}={divergence.expected}, "
        f"rank {divergence.divergent_rank}={divergence.actual}"
    )


class NcclCollectiveRecorder:
    """Record completed work and sequence counters without adding collectives."""

    def __init__(self, groups: Mapping[str, ProcessGroup]) -> None:
        self._groups = dict(groups)
        self._initial_sequences = {
            name: int(group._get_sequence_number_for_group()) for name, group in self._groups.items()
        }
        self._events: dict[str, list[CollectiveEvent]] = {name: [] for name in self._groups}
        self._boundaries: list[CollectiveBoundary] = []
        self._errors: list[str] = []
        self._condition = threading.Condition()

        for name, group in self._groups.items():
            group._register_on_completion_hook(self._make_hook(name))

    def _make_hook(self, group_name: str):
        def record(work_info) -> None:
            try:
                operation = getattr(work_info.op_type, "name", str(work_info.op_type))
                event = CollectiveEvent(str(operation), int(work_info.seq))
                with self._condition:
                    self._events[group_name].append(event)
                    self._condition.notify_all()
            except Exception as error:  # hooks run on the NCCL watchdog thread
                with self._condition:
                    self._errors.append(f"{group_name}: {error!r}")
                    self._condition.notify_all()

        return record

    def boundary(self, label: str) -> None:
        sequences = {
            name: int(group._get_sequence_number_for_group()) - self._initial_sequences[name]
            for name, group in self._groups.items()
        }
        with self._condition:
            self._boundaries.append(CollectiveBoundary(label, sequences))

    def finish(
        self,
        *,
        rank: int,
        mesh_dim_names: Sequence[str],
        mesh_shape: Sequence[int],
        mesh_coordinate: Sequence[int],
        timeout_seconds: float = 30,
    ) -> RankCollectiveSchedule:
        expected_counts = {
            name: int(group._get_sequence_number_for_group()) - self._initial_sequences[name]
            for name, group in self._groups.items()
        }
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not self._errors and any(
                len(self._events[name]) < expected_count for name, expected_count in expected_counts.items()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    observed = {name: len(events) for name, events in self._events.items()}
                    raise TimeoutError(
                        f"NCCL completion hooks did not observe every collective; "
                        f"expected={expected_counts}, observed={observed}"
                    )
                self._condition.wait(timeout=remaining)
            if self._errors:
                raise RuntimeError(f"NCCL completion hook failed: {self._errors}")
            events = {
                name: tuple(sorted(group_events, key=lambda event: event.sequence_number))
                for name, group_events in self._events.items()
            }
            boundaries = tuple(self._boundaries)

        return RankCollectiveSchedule(
            rank=rank,
            mesh_dim_names=tuple(mesh_dim_names),
            mesh_shape=tuple(mesh_shape),
            mesh_coordinate=tuple(mesh_coordinate),
            events=events,
            boundaries=boundaries,
        )
