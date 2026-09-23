"""Wall time and waits for each trajectory-runner call.

A ContextVar follows child coroutines, and each runner call owns its accumulator.
Collect, assemble and finalize partition the call's wall time. Wait totals can
exceed it and are emitted separately. Publishing only enqueues records.
"""

import asyncio
import contextlib
import json
import math
import time
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import Executor
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
from uuid import uuid4

from skyrl_train.telemetry import (
    TRAINER_ROLE,
    phase_attributes,
    phase_duration,
    record_event,
    run_in_executor_observed,
    telemetry,
)


RolloutPhase = Literal["collect", "assemble", "finalize", "tokenize", "retain"]
_PARENTS = {
    "collect": "rollout_call",
    "assemble": "rollout_call",
    "finalize": "rollout_call",
    "tokenize": "collect",
    "retain": "finalize",
}
_EXCLUSIVE_PHASES = ("collect", "assemble", "finalize")
# At most 64 finite float pairs fit below the exporter's 4096-byte string limit. Each pair is
# the start and end of one model await in seconds since the call started.
_MAX_MODEL_INTERVALS = 64
_CURRENT: ContextVar["RolloutObservation | None"] = ContextVar("rollout_observation", default=None)
wait_seconds = telemetry.histogram("rollout_wait_seconds", unit="s")
waits = telemetry.counter("rollout_waits", unit="{wait}")
calls = telemetry.counter("rollout_calls", unit="{call}")
buffer_dwell = telemetry.histogram("rollout_buffer_dwell_seconds", unit="s")
groups = telemetry.counter("rollout_groups", unit="{group}")
group_tokens = telemetry.counter("rollout_group_tokens", unit="{token}")
event_loop_lag = telemetry.histogram("event_loop_lag_seconds", unit="s")


@contextlib.contextmanager
def async_phase_window(phase: str, *, step: int, enabled: bool) -> Iterator[None]:
    """Record a driver phase's wall-clock window for joins with sampled engine counters.

    The Unix timestamps join with imported metric snapshots, and the monotonic
    duration detects clock adjustments. The window does not measure GPU execution.
    """
    if not enabled:
        yield
        return
    started_unix_ms = time.time_ns() // 1_000_000
    started = time.perf_counter()
    outcome = "success"
    try:
        yield
    except BaseException:
        outcome = "failure"
        raise
    finally:
        record_event(
            "async_phase_window",
            {
                "started_unix_ms": started_unix_ms,
                "finished_unix_ms": time.time_ns() // 1_000_000,
                "duration_seconds": time.perf_counter() - started,
            },
            attributes={"phase": phase, "step": str(step), "role": TRAINER_ROLE, "outcome": outcome},
        )


async def monitor_event_loop_lag(
    *,
    step_fn: Callable[[], int],
    interval: float = 1.0,
    clock: Callable[[], float] = time.perf_counter,
    wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Sample driver scheduling delay until cancelled, without replaying missed ticks."""
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("event-loop lag sampling interval must be finite and positive")
    while True:
        expected = clock() + interval
        await wait(interval)
        event_loop_lag.record(
            max(0.0, clock() - expected),
            attributes={"role": TRAINER_ROLE, "step": str(step_fn()), "mode": "async"},
        )


@dataclass
class WaitObservation:
    total: float = 0.0
    count: int = 0
    maximum: float = 0.0

    def add(self, duration: float) -> None:
        self.total += duration
        self.count += 1
        self.maximum = max(self.maximum, duration)


def publish_wait(name: str, wait: WaitObservation, *, step: int, mode: str) -> None:
    attributes = {"wait": name, "role": TRAINER_ROLE, "step": str(step), "mode": mode}
    wait_seconds.record(wait.total, attributes={**attributes, "stat": "sum"})
    wait_seconds.record(wait.maximum, attributes={**attributes, "stat": "max"})
    waits.add(wait.count, attributes=attributes)


@dataclass
class RolloutObservation:
    step: int
    mode: str
    clock: Callable[[], float] = time.perf_counter
    started: float = 0.0
    started_unix_ms: int = 0
    call_id: str = field(default_factory=lambda: uuid4().hex)
    durations: dict[str, float] = field(default_factory=dict)
    waits: dict[str, WaitObservation] = field(default_factory=dict)
    model_awaits: list[tuple[float, float]] = field(default_factory=list)
    model_await_count: int = 0
    response_tokens: int = 0

    def record_wait(self, name: str, duration: float) -> None:
        self.waits.setdefault(name, WaitObservation()).add(duration)

    def publish(self, finished: float, outcome: str) -> None:
        total = finished - self.started
        residual = total - sum(self.durations.get(name, 0.0) for name in _EXCLUSIVE_PHASES)
        attributes = {"role": TRAINER_ROLE, "step": str(self.step), "mode": self.mode, "outcome": outcome}
        calls.add(1, attributes=attributes)
        phases = {"rollout_call": total, **self.durations, "rollout_call_residual": residual}
        for name, duration in phases.items():
            parent = _PARENTS.get(name, "rollout_call" if name == "rollout_call_residual" else None)
            phase_duration.record(
                duration,
                attributes={
                    **attributes,
                    **phase_attributes(
                        phase=f"rollout_{name}" if name in _PARENTS else name,
                        root="rollout_call",
                        parent=None if parent is None else (f"rollout_{parent}" if parent in _PARENTS else parent),
                        clock_domain="driver_monotonic",
                    ),
                },
            )
        for name, wait in self.waits.items():
            publish_wait(name, wait, step=self.step, mode=self.mode)
        # The bounded metric attributes above do not contain per-call identities.
        # A single event retains the complete accounting needed to audit one call.
        record_event(
            "rollout_call",
            {
                "call_id": self.call_id,
                "started_unix_ms": self.started_unix_ms,
                "finished_unix_ms": time.time_ns() // 1_000_000,
                "duration_seconds": total,
                **{f"duration_{name}": duration for name, duration in phases.items() if name != "rollout_call"},
                "model_awaits_json": json.dumps(self.model_awaits, separators=(",", ":"), allow_nan=False),
                "interval_count": self.model_await_count,
                "truncated": self.model_await_count > len(self.model_awaits),
                "response_tokens": self.response_tokens,
            },
            attributes=attributes,
        )


@contextlib.contextmanager
def observe_rollout_call(
    *, step: int, mode: str, enabled: bool, clock: Callable[[], float] = time.perf_counter
) -> Iterator[RolloutObservation | None]:
    if not enabled:
        yield None
        return
    observation = RolloutObservation(
        step=step, mode=mode, clock=clock, started=clock(), started_unix_ms=time.time_ns() // 1_000_000
    )
    token = _CURRENT.set(observation)
    outcome = "success"
    try:
        yield observation
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException:
        outcome = "failure"
        raise
    finally:
        finished = clock()
        _CURRENT.reset(token)
        observation.publish(finished, outcome)


@contextlib.contextmanager
def rollout_phase(name: RolloutPhase) -> Iterator[None]:
    observation = _CURRENT.get()
    if observation is None:
        yield
        return
    started = observation.clock()
    try:
        yield
    finally:
        observation.durations[name] = observation.durations.get(name, 0.0) + observation.clock() - started


@contextlib.contextmanager
def rollout_wait(name: str) -> Iterator[None]:
    observation = _CURRENT.get()
    if observation is None:
        yield
        return
    started = observation.clock()
    try:
        yield
    finally:
        finished = observation.clock()
        observation.record_wait(name, finished - started)
        if name == "model_client_await":
            observation.model_await_count += 1
            if len(observation.model_awaits) < _MAX_MODEL_INTERVALS:
                observation.model_awaits.append((started - observation.started, finished - observation.started))


def time_tokenization(func: Callable, *args, **kwargs):
    with rollout_phase("tokenize"):
        return func(*args, **kwargs)


@contextlib.contextmanager
def async_wait(name: str, *, step: int, enabled: bool) -> Iterator[None]:
    """Measure a producer await outside the trajectory-runner call."""
    if not enabled:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        publish_wait(name, WaitObservation(elapsed, 1, elapsed), step=step, mode="async")


async def run_environment(executor: Executor | None, func: Callable, *args, **kwargs):
    """Run ``func`` on ``executor`` (inline when None) and return its result, recording queue, execution and driver-resume delays."""
    observation = _CURRENT.get()
    call = partial(func, *args, **kwargs)
    if observation is None:
        return call() if executor is None else await run_in_executor_observed(executor, "environment", call)
    clock = observation.clock
    submitted = clock()
    stamps: list[float] = []

    def invoke():
        stamps.append(clock())
        try:
            return call()
        finally:
            stamps.append(clock())

    try:
        with rollout_wait("env_await"):
            return invoke() if executor is None else await run_in_executor_observed(executor, "environment", invoke)
    finally:
        # Cancellation can leave the executor running. Do not invent an execution
        # duration or mutate a published accumulator when that thread later exits.
        if len(stamps) == 2:
            observation.record_wait("env_queue", stamps[0] - submitted)
            observation.record_wait("env_exec", stamps[1] - stamps[0])
            observation.record_wait("env_resume", clock() - stamps[1])


def record_group_disposition(
    *,
    disposition: str,
    tokens: int,
    step: int,
    completed_at: float | None = None,
    call_id: str | None = None,
    admitted_at: float | None = None,
) -> None:
    attributes = {"role": TRAINER_ROLE, "step": str(step), "disposition": disposition}
    groups.add(1, attributes=attributes)
    group_tokens.add(tokens, attributes=attributes)
    if completed_at is not None:
        finished = time.perf_counter() if admitted_at is None else admitted_at
        buffer_dwell.record(finished - completed_at, attributes=attributes)
    if call_id is not None:
        record_event("rollout_group_disposition", {"call_id": call_id, "tokens": tokens}, attributes=attributes)
