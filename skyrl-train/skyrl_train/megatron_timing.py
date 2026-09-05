"""Bounded timing decomposition for Megatron policy updates."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from skyrl_train.telemetry import WORKER_ROLE, phase_duration


FORWARD_BACKWARD_SCHEDULER = "megatron_forward_backward_scheduler"
PIPELINE_METRIC_BROADCAST = "megatron_pipeline_metric_broadcast"
OPTIMIZER_STEP = "megatron_optimizer_step"
WORLD_METRIC_REDUCTION = "megatron_world_metric_reduction"
FINAL_BARRIER = "megatron_final_barrier"
TOTAL = "megatron_policy_train_total"
RESIDUAL = "megatron_policy_train_residual"

EXCLUSIVE_PHASES = frozenset(
    {
        FORWARD_BACKWARD_SCHEDULER,
        PIPELINE_METRIC_BROADCAST,
        OPTIMIZER_STEP,
        WORLD_METRIC_REDUCTION,
        FINAL_BARRIER,
    }
)


@dataclass(frozen=True)
class MegatronTimingObservation:
    """One worker-local timing observation from a Megatron policy update."""

    phase: str
    seconds: float
    parent_phase: str | None


class DurationRecorder(Protocol):
    def record(self, value: float, *, attributes: dict[str, str]) -> None: ...


class MegatronTrainTimings:
    """Accumulate worker-local CPU dispatch wall without synchronizing pipeline phases.

    These measurements intentionally do not insert CUDA synchronization or pipeline
    barriers. The existing final distributed barrier provides the enclosing completion
    boundary, while individual phases describe host-observed dispatch/blocking wall.
    """

    def __init__(self, *, enabled: bool, clock: Callable[[], float] = time.perf_counter) -> None:
        self.enabled = enabled
        self._clock = clock
        self._started = clock() if enabled else None
        self._durations = {phase: 0.0 for phase in EXCLUSIVE_PHASES}
        self._finished = False

    @contextmanager
    def span(self, phase: str) -> Iterator[None]:
        """Accumulate one occurrence of a declared exclusive phase."""
        if phase not in EXCLUSIVE_PHASES:
            raise ValueError(f"unknown Megatron timing phase: {phase!r}")
        if not self.enabled:
            yield
            return

        started = self._clock()
        try:
            yield
        finally:
            self._durations[phase] += self._clock() - started

    def finish(self) -> tuple[MegatronTimingObservation, ...]:
        """Close the total and return leaf, total, and signed residual observations."""
        if not self.enabled:
            return ()
        if self._finished:
            raise RuntimeError("Megatron policy timing was already finished")
        self._finished = True
        assert self._started is not None
        total = self._clock() - self._started
        covered = sum(self._durations.values())
        observations = [
            MegatronTimingObservation(phase=phase, seconds=self._durations[phase], parent_phase=TOTAL)
            for phase in sorted(EXCLUSIVE_PHASES)
        ]
        observations.extend(
            (
                MegatronTimingObservation(phase=RESIDUAL, seconds=total - covered, parent_phase=TOTAL),
                MegatronTimingObservation(phase=TOTAL, seconds=total, parent_phase=None),
            )
        )
        return tuple(observations)


def publish_megatron_train_timings(
    observations: Sequence[MegatronTimingObservation],
    *,
    step: int,
    rank: int,
    outcome: str,
    recorder: DurationRecorder = phase_duration,
) -> None:
    """Enqueue worker timings without flushing the process telemetry exporter."""
    base_attributes = {
        "backend": "megatron",
        "clock_domain": "cpu_dispatch_wall",
        "outcome": outcome,
        "rank": str(rank),
        "role": WORKER_ROLE,
        "step": str(step),
        "root": TOTAL,
    }
    for observation in observations:
        attributes = {**base_attributes, "phase": observation.phase}
        if observation.parent_phase is not None:
            attributes["parent"] = observation.parent_phase
        recorder.record(observation.seconds, attributes=attributes)
