"""Low-cardinality telemetry for Harbor lifecycle state and result timings."""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

from skyrl_train.telemetry import record_telemetry_health, telemetry


HEARTBEAT_SECONDS = 30.0

transition_count = telemetry.counter("harbor_lifecycle_transitions", unit="{transition}")
stage_work = telemetry.gauge("harbor_stage_work", unit="{trial}")
pending_age = telemetry.gauge("harbor_oldest_pending_age_seconds", unit="s")
last_progress_age = telemetry.gauge("harbor_last_progress_age_seconds", unit="s")
trial_phase_duration = telemetry.histogram("harbor_trial_phase_duration_seconds", unit="s")
trial_results = telemetry.counter("harbor_trial_results", unit="{trial}")
trial_retries = telemetry.counter("harbor_trial_retries", unit="{trial}")
trial_tokens = telemetry.histogram("harbor_trial_tokens", unit="{token}")
group_tail = telemetry.histogram("harbor_group_tail_seconds", unit="s")
dispatch_groups = telemetry.counter("harbor_dispatch_groups", unit="{group}")
dispatch_duration = telemetry.histogram("harbor_dispatch_duration_seconds", unit="s")


@dataclass
class _TrialState:
    stage: str
    started: float
    last_progress: float


def _duration(timing: Any) -> float | None:
    if timing is None or timing.started_at is None or timing.finished_at is None:
        return None
    return max(0.0, (timing.finished_at - timing.started_at).total_seconds())


def _result_durations(result: Any) -> dict[str, float]:
    phases: dict[str, float] = {}
    for name in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
        if (duration := _duration(getattr(result, name, None))) is not None:
            phases[name] = duration
    started_at = getattr(result, "started_at", None)
    finished_at = getattr(result, "finished_at", None)
    if started_at is not None and finished_at is not None:
        phases["total"] = max(0.0, (finished_at - started_at).total_seconds())
    return phases


def record_harbor_group(results: Iterable[Any]) -> None:
    """Aggregate result tails after the all-trials barrier without trial IDs as labels."""
    totals: list[float] = []
    slowest_phases: list[tuple[float, str]] = []
    for result in results:
        if isinstance(result, BaseException):
            continue
        phases = _result_durations(result)
        if "total" in phases:
            totals.append(phases["total"])
        phase_items = [(duration, phase) for phase, duration in phases.items() if phase != "total"]
        if phase_items:
            slowest_phases.append(max(phase_items))
    if not totals:
        return
    for statistic, value in (
        ("p50", statistics.median(totals)),
        ("max", max(totals)),
        ("spread", max(totals) - min(totals)),
    ):
        group_tail.record(value, attributes={"statistic": statistic, "slowest_phase": "none"})
    if slowest_phases:
        duration, phase = max(slowest_phases)
        group_tail.record(duration, attributes={"statistic": "slowest_phase", "slowest_phase": phase})


def record_dispatcher_heartbeat(pending: list[int], last_progress: list[float | None], now: float) -> None:
    stage_work.set(sum(pending), attributes={"stage": "dispatcher_pending"})
    active_ages = [now - value for value, count in zip(last_progress, pending, strict=True) if value is not None and count]
    last_progress_age.set(max(active_ages, default=0.0), attributes={"subsystem": "harbor_dispatcher"})
    record_telemetry_health()


def record_dispatch_group(duration_seconds: float, outcome: str) -> None:
    attributes = {"outcome": outcome, "subsystem": "harbor_dispatcher"}
    dispatch_groups.add(1, attributes=attributes)
    dispatch_duration.record(duration_seconds, attributes=attributes)


class HarborLifecycleObserver:
    """Track lifecycle transitions, pending age, and existing TrialResult timings."""

    def __init__(self, heartbeat_seconds: float = HEARTBEAT_SECONDS) -> None:
        self._heartbeat_seconds = heartbeat_seconds
        self._submitted: deque[float] = deque()
        self._trials: dict[str, _TrialState] = {}
        self._stage_counts: Counter[str] = Counter()
        self._recorded_results: set[str] = set()
        self._last_exception: dict[str, str] = {}
        self._last_progress = time.monotonic()
        self._heartbeat_task: asyncio.Task | None = None

    def start(self) -> None:
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.heartbeat()

    def register_submitted(self, count: int) -> None:
        now = time.monotonic()
        self._submitted.extend([now] * count)
        self._last_progress = now
        self.heartbeat(now)

    def discard_submitted(self, count: int) -> None:
        for _ in range(min(count, len(self._submitted))):
            self._submitted.pop()
        self._last_progress = time.monotonic()
        self.heartbeat()

    async def __call__(self, event: Any) -> None:
        now = time.monotonic()
        raw_event = getattr(event, "event", None)
        stage = getattr(raw_event, "value", None) or "end"
        result = getattr(event, "result", None)
        if result is None and not hasattr(event, "event"):
            result = event
        trial_id = str(getattr(event, "trial_id", getattr(result, "id", "unknown")))
        task_name = str(getattr(event, "task_name", getattr(result, "task_name", "unknown")))

        if stage == "start" and trial_id in self._recorded_results:
            self._recorded_results.remove(trial_id)
            reason = self._last_exception.pop(trial_id, None)
            if reason is not None:
                trial_retries.add(1, attributes={"reason": reason, "subsystem": "harbor"})
                transition_count.add(1, attributes={"stage": "retry", "subsystem": "harbor"})

        previous = self._trials.get(trial_id)
        if previous is not None:
            self._stage_counts[previous.stage] -= 1
            stage_work.set(self._stage_counts[previous.stage], attributes={"stage": previous.stage})
        if stage == "start" and self._submitted:
            self._submitted.popleft()

        terminal = stage in {"end", "cancel"}
        if terminal:
            self._trials.pop(trial_id, None)
        else:
            started = previous.started if previous is not None else now
            self._trials[trial_id] = _TrialState(stage=stage, started=started, last_progress=now)
            self._stage_counts[stage] += 1
            stage_work.set(self._stage_counts[stage], attributes={"stage": stage})

        self._last_progress = now
        transition_count.add(1, attributes={"stage": stage, "subsystem": "harbor"})
        logger.bind(
            subsystem="harbor",
            trial_id=trial_id,
            task_name=task_name,
            lifecycle_stage=stage,
        ).info("Harbor lifecycle transition")
        if result is not None and trial_id not in self._recorded_results:
            self._record_result(trial_id, result)
        self.heartbeat(now)

    def _record_result(self, trial_id: str, result: Any) -> None:
        self._recorded_results.add(trial_id)
        exception = getattr(getattr(result, "exception_info", None), "exception_type", None) or "none"
        outcome = "failure" if exception != "none" else "success"
        if exception != "none":
            self._last_exception[trial_id] = str(exception)
        trial_results.add(1, attributes={"outcome": outcome, "exception_class": str(exception)})
        for phase, duration in _result_durations(result).items():
            trial_phase_duration.record(duration, attributes={"phase": phase, "outcome": outcome})
        compute_tokens = getattr(result, "compute_token_cost_totals", None)
        if compute_tokens is not None:
            input_tokens, cache_tokens, output_tokens, _ = compute_tokens()
            for token_kind, value in (
                ("input", input_tokens),
                ("cache", cache_tokens),
                ("output", output_tokens),
            ):
                if value is not None:
                    trial_tokens.record(value, attributes={"token_kind": token_kind})

    def snapshot(self, now: float | None = None) -> dict[str, float | int]:
        now = time.monotonic() if now is None else now
        ages = [now - submitted for submitted in self._submitted]
        ages.extend(now - trial.started for trial in self._trials.values())
        return {
            "pending": len(self._submitted) + len(self._trials),
            "queued": len(self._submitted),
            "active": len(self._trials),
            "oldest_pending_age_seconds": max(ages, default=0.0),
            "last_progress_age_seconds": max(0.0, now - self._last_progress),
        }

    def heartbeat(self, now: float | None = None) -> None:
        snapshot = self.snapshot(now)
        for state in ("pending", "queued", "active"):
            stage_work.set(snapshot[state], attributes={"stage": state})
        pending_age.set(snapshot["oldest_pending_age_seconds"], attributes={"subsystem": "harbor"})
        last_progress_age.set(snapshot["last_progress_age_seconds"], attributes={"subsystem": "harbor"})
        record_telemetry_health()

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            if self._submitted or self._trials:
                self.heartbeat()
