"""Per-call rollout walls and concurrent waits, without step-wall folding.

The ContextVar follows child coroutines but each runner call owns its accumulator.
Only collect, assemble and finalize partition the call wall. Wait totals can
exceed that wall and are emitted separately. Publishing only enqueues records.
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

from skyrl_train.telemetry import TRAINER_ROLE, phase_duration, record_event, telemetry


RolloutPhase = Literal["collect", "assemble", "finalize", "tokenize", "retain"]
_PARENTS = {
    "collect": "rollout_call",
    "assemble": "rollout_call",
    "finalize": "rollout_call",
    "tokenize": "collect",
    "retain": "finalize",
}
_EXCLUSIVE_PHASES = ("collect", "assemble", "finalize")
# At most 64 finite float pairs fit below the exporter's 4096-byte string limit.
_MAX_MODEL_INTERVALS = 64
_CURRENT: ContextVar["RolloutObservation | None"] = ContextVar("rollout_observation", default=None)
wait_seconds = telemetry.histogram("rollout_wait_seconds", unit="s")
wait_count = telemetry.counter("rollout_wait_count", unit="{wait}")
call_count = telemetry.counter("rollout_call_count", unit="{call}")
buffer_dwell = telemetry.histogram("rollout_buffer_dwell_seconds", unit="s")
group_count = telemetry.counter("rollout_group_count", unit="{group}")
group_tokens = telemetry.counter("rollout_group_tokens", unit="{token}")
event_loop_lag = telemetry.histogram("event_loop_lag_seconds", unit="s")


def async_step_metrics(
    *,
    core_seconds: float,
    cycle_seconds: float,
    buffer_wait_seconds: float,
    training_seconds: float,
    sync_seconds: float,
    consumed_loss_tokens: int,
    consumed_response_tokens: int,
    policy_gpus: int,
    inference_gpus: int,
) -> dict[str, float]:
    """Summarize driver walls and useful work; GPU denominators are configured roles.

    Core excludes callbacks, checkpoint and evaluation. Cycle includes them up to
    metric publication; neither includes startup, inter-epoch cleanup or final
    export. These rates are not whole-job billed efficiency or GPU utilization.
    """
    metrics = {
        "core_seconds": core_seconds,
        "cycle_seconds": cycle_seconds,
        "outside_core_seconds": cycle_seconds - core_seconds,
        "configured_policy_gpus": float(policy_gpus),
        "configured_inference_gpus": float(inference_gpus),
        "consumed_loss_tokens": float(consumed_loss_tokens),
        "consumed_response_tokens": float(consumed_response_tokens),
    }
    if core_seconds > 0:
        metrics.update(
            buffer_wait_fraction=buffer_wait_seconds / core_seconds,
            training_fraction=training_seconds / core_seconds,
            weight_sync_fraction=sync_seconds / core_seconds,
            consumed_loss_tokens_per_core_second=consumed_loss_tokens / core_seconds,
        )
    if cycle_seconds > 0:
        metrics["consumed_loss_tokens_per_cycle_second"] = consumed_loss_tokens / cycle_seconds
        if policy_gpus > 0:
            metrics["loss_tokens_per_configured_policy_gpu_second"] = consumed_loss_tokens / cycle_seconds / policy_gpus
        if inference_gpus > 0:
            metrics["response_tokens_per_configured_inference_gpu_second"] = (
                consumed_response_tokens / cycle_seconds / inference_gpus
            )
    return {f"async/performance/{name}": value for name, value in metrics.items()}


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
    wait_count.add(wait.count, attributes=attributes)


@dataclass
class RolloutObservation:
    step: int
    mode: str
    clock: Callable[[], float] = time.perf_counter
    call_id: str = field(default_factory=lambda: uuid4().hex)
    durations: dict[str, float] = field(default_factory=dict)
    waits: dict[str, WaitObservation] = field(default_factory=dict)
    model_awaits: list[tuple[float, float]] = field(default_factory=list)
    model_await_count: int = 0
    response_tokens: int = 0

    def record_wait(self, name: str, duration: float) -> None:
        self.waits.setdefault(name, WaitObservation()).add(duration)

    def publish(self, started: float, finished: float, outcome: str) -> None:
        total = finished - started
        residual = total - sum(self.durations.get(name, 0.0) for name in _EXCLUSIVE_PHASES)
        attributes = {"role": TRAINER_ROLE, "step": str(self.step), "mode": self.mode, "outcome": outcome}
        call_count.add(1, attributes=attributes)
        phases = {"rollout_call": total, **self.durations, "rollout_call_residual": residual}
        for name, duration in phases.items():
            parent = _PARENTS.get(name, "rollout_call" if name == "rollout_call_residual" else None)
            phase_duration.record(
                duration,
                attributes={
                    **attributes,
                    "phase": f"rollout_{name}" if name in _PARENTS else name,
                    "root": "rollout_call",
                    **({"parent": f"rollout_{parent}" if parent in _PARENTS else parent} if parent is not None else {}),
                    "clock_domain": "driver_monotonic",
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
                "started": started,
                "finished": finished,
                **{f"duration_{name}": duration for name, duration in phases.items()},
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
    observation = RolloutObservation(step=step, mode=mode, clock=clock)
    token = _CURRENT.set(observation)
    started = clock()
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
        observation.publish(started, finished, outcome)


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
                observation.model_awaits.append((started, finished))


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
    """Separate executor queue, execution and driver-resume delay."""
    observation = _CURRENT.get()
    call = partial(func, *args, **kwargs)
    if observation is None:
        return call() if executor is None else await asyncio.get_running_loop().run_in_executor(executor, call)
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
            return invoke() if executor is None else await asyncio.get_running_loop().run_in_executor(executor, invoke)
    finally:
        # Cancellation can leave the executor running. Do not invent an execution
        # duration or mutate a published accumulator when that thread later exits.
        if len(stamps) == 2:
            observation.record_wait("env_queue", stamps[0] - submitted)
            observation.record_wait("env_exec", stamps[1] - stamps[0])
            observation.record_wait("env_resume", clock() - stamps[1])


def record_group_outcome(
    *,
    outcome: str,
    tokens: int,
    step: int,
    completed_at: float | None = None,
    attempt_id: str | None = None,
    admitted_at: float | None = None,
) -> None:
    attributes = {"role": TRAINER_ROLE, "step": str(step), "outcome": outcome}
    group_count.add(1, attributes=attributes)
    group_tokens.add(tokens, attributes=attributes)
    if completed_at is not None:
        finished = time.perf_counter() if admitted_at is None else admitted_at
        buffer_dwell.record(finished - completed_at, attributes=attributes)
    if attempt_id is not None:
        record_event("rollout_group_outcome", {"call_id": attempt_id, "tokens": tokens}, attributes=attributes)
