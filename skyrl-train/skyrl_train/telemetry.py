import asyncio
import contextlib
import functools
import math
import os
import socket
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

import ray
from loguru import logger

from marinskyrl.environment_contract import TRAINING_LOOP_ENV, TrainingLoop

try:
    from rigging import telemetry
    from rigging.telemetry.serialization import EventBody
except ImportError as error:
    # An installed rigging without the telemetry submodule raises ImportError, not
    # ModuleNotFoundError; the name check still keeps a failure inside rigging visible.
    if error.name != "rigging":
        raise
    from skyrl_train import inert_telemetry as telemetry
    from skyrl_train.inert_telemetry import EventBody


# A process that forwards a foreign system's metrics publishes under that system's name; this one
# covers everything MarinSkyRL measures about itself.
SERVICE = "marinskyrl"
DRIVER_ROLE = "driver"
TRAINER_ROLE = "trainer"
CONTROLLER_ROLE = "controller"
WORKER_ROLE = "worker"
HARBOR_COORDINATOR_ROLE = "harbor_coordinator"
SHUTDOWN_TIMEOUT_SECONDS = 2.0


class StepKind(StrEnum):
    """Which step counter a record's `step` attribute counts.

    These members are exported verbatim as the `step_kind` attribute value, so the
    strings are part of the published schema: dashboards filter on them and renaming
    a member's value breaks every panel keyed to it.
    """

    GLOBAL_STEP = "global_step"
    MODEL_VERSION_STEP = "model_version_step"
    # No step was supplied, so neither counter describes the record.
    UNKNOWN = "unknown"


work_completed = telemetry.counter("work_completed", unit="{item}")
phase_duration = telemetry.histogram("phase_duration_seconds", unit="s")
# rigging publishes `queue_depth` and `progress_time_seconds` for its own exporter from the process
# that configures the service, so a plain name here lands its rows in our namespace -- on a real run
# it outnumbered ours under `progress_time_seconds` by more than two orders of magnitude. Hence the
# `rollout_` prefix below. `progress_time_seconds` keeps the plain name because levanter writes it
# and finelog's TRAINING_STATUS_NAMES reads it; ours carry `work_kind`, rigging's `progress_kind`.
progress_timestamp = telemetry.gauge("progress_time_seconds", unit="s")
policy_step = telemetry.gauge("policy_step", unit="{step}")
rollout_queue_depth = telemetry.gauge("rollout_queue_depth", unit="{item}")
rollout_capacity = telemetry.gauge("rollout_capacity", unit="{item}")
rollout_staleness = telemetry.histogram("rollout_staleness_steps", unit="{step}")
training_metric = telemetry.histogram("training_metric_value")
training_nonfinite_values = telemetry.counter("training_nonfinite_values", unit="{value}")
generation_group_duration = telemetry.histogram("generation_group_duration_seconds", unit="s")
generation_groups = telemetry.counter("generation_groups", unit="{group}")
generation_active_groups = telemetry.gauge("generation_active_groups", unit="{group}")
generation_input_coverage = telemetry.counter("generation_input_coverage", unit="{group}")
admission_groups = telemetry.counter("admission_groups", unit="{group}")
executor_queue_delay = telemetry.histogram("executor_queue_delay_seconds", unit="s")
executor_duration = telemetry.histogram("executor_duration_seconds", unit="s")
executor_work = telemetry.gauge("executor_work", unit="{item}")
executor_cancellations = telemetry.counter("executor_cancellations", unit="{item}")
judge_request_duration = telemetry.histogram("judge_request_duration_seconds", unit="s")
judge_requests = telemetry.counter("judge_requests", unit="{request}")
judge_parse_events = telemetry.counter("judge_parse_events", unit="{event}")
judge_cohort_duration = telemetry.histogram("judge_cohort_duration_seconds", unit="s")
telemetry_smoke = telemetry.gauge("telemetry_smoke", unit="1")
generation_retained_groups = telemetry.gauge("generation_retained_groups", unit="{group}")
generation_retained_rows = telemetry.gauge("generation_retained_rows", unit="{row}")
generation_retained_estimated_bytes = telemetry.gauge("generation_retained_estimated_bytes", unit="By")
generation_producers = telemetry.gauge("generation_producers", unit="{producer}")
generation_retention_events = telemetry.gauge("generation_retention_events", unit="{event}")
process_memory_bytes = telemetry.gauge("process_memory_bytes", unit="By")
generation_memory_boundaries = telemetry.counter("generation_memory_boundaries", unit="{boundary}")

_RETENTION_OWNERS = ("producer", "completed_buffer", "admission", "admitted")
_PRODUCER_STATES = ("waiting_input", "generating", "projecting", "holding_completed", "blocked_on_buffer")
_RETENTION_FIELDS = (
    "prompt_token_ids",
    "response_ids",
    "loss_masks",
    "rollout_logprobs",
    "student_topk_indices",
    "behavior_topk_logprobs",
    "rollout_routed_experts",
    "source_prompts",
    "other",
)

_generation_active: defaultdict[str, int] = defaultdict(int)
_executor_counts: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"queued": 0, "active": 0})
_executor_lock = threading.Lock()


def record_event(
    name: str,
    fields: dict[str, str | int | float | bool | None],
    *,
    attributes: dict[str, str] | None = None,
) -> None:
    """Enqueue a flat event, omitting unavailable values."""
    telemetry.event(
        name, EventBody({key: value for key, value in fields.items() if value is not None}), attributes=attributes
    )


@dataclass(frozen=True)
class ConsumedWork:
    """Rows and tokens one optimizer step consumed, excluding data-parallel padding."""

    sequences: int
    response_tokens: int
    loss_tokens: int


def record_consumed_work(work: ConsumedWork, *, step: int) -> None:
    """Record useful work after an optimizer step completes successfully."""
    attributes = {"role": TRAINER_ROLE, "step": str(step)}
    for kind, count in (
        ("consumed_sample", work.sequences),
        ("consumed_response_token", work.response_tokens),
        ("consumed_loss_token", work.loss_tokens),
    ):
        work_completed.add(count, attributes={**attributes, "work_kind": kind})


def record_training_metrics(metrics: Mapping[str, object], *, step: int, kind: str) -> None:
    """Mirror finite numeric trainer scalars without changing their values."""
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        attributes = {"metric": name, "step": str(step), "role": TRAINER_ROLE, "payload_kind": kind}
        if math.isfinite(value):
            training_metric.record(float(value), attributes=attributes)
        else:
            training_nonfinite_values.add(1, attributes=attributes)


class _BackgroundCollector(Protocol):
    def start(self) -> None: ...

    def stop(self, *, timeout: float) -> None: ...


class _InertCollector:
    def start(self) -> None:
        pass

    def stop(self, *, timeout: float) -> None:
        pass


_inert_collector = _InertCollector()


@dataclass
class _ProcessState:
    owner: "ProcessTelemetry | None" = None
    policy_step: int | None = None
    last_progress_timestamp: float | None = None
    queue_depth: int | None = None
    queue_capacity: int | None = None

    def claim(self, owner: "ProcessTelemetry") -> bool:
        if self.owner is not None:
            return False
        self.owner = owner
        self.policy_step = None
        self.last_progress_timestamp = None
        self.queue_depth = None
        self.queue_capacity = None
        return True

    def release(self, owner: "ProcessTelemetry") -> None:
        if self.owner is owner:
            self.owner = None


# Rigging has one process-wide runtime, so application progress follows its single lifecycle owner.
_process_state = _ProcessState()


@dataclass(frozen=True)
class TelemetryConfig:
    endpoint: str | None = None
    run_id: str | None = None
    execution_uid: str | None = None
    serving_job_id: str | None = None
    training_loop: TrainingLoop | None = None

    @classmethod
    def from_environment(cls) -> "TelemetryConfig":
        def text(environment_name: str) -> str | None:
            value = os.environ.get(environment_name, "").strip()
            return value or None

        return cls(
            endpoint=text("SKYRL_TELEMETRY_ENDPOINT"),
            run_id=text("SKYRL_RUN_ID"),
            execution_uid=text("SKYRL_EXECUTION_UID") or _iris_execution_uid(),
            serving_job_id=text("SKYRL_SERVING_JOB_ID"),
            training_loop=TrainingLoop(loop) if (loop := text(TRAINING_LOOP_ENV)) else None,
        )


def _iris_execution_uid() -> str | None:
    if attempt_uid := os.environ.get("IRIS_ATTEMPT_UID"):
        return f"iris:{attempt_uid}"
    return None


def _iris_resources() -> dict[str, str]:
    task_with_attempt = os.environ.get("IRIS_TASK_ID")
    resources: dict[str, str] = {}
    if task_with_attempt:
        task_id, separator, attempt = task_with_attempt.rpartition(":")
        if not separator or "/" in attempt:
            task_id, attempt = task_with_attempt, ""
        job_id, separator, _ = task_id.rpartition("/")
        if separator:
            resources.update(job_id=job_id, task_id=task_id)
        if attempt:
            resources["attempt"] = attempt
    for environment_name, resource_name in (
        ("IRIS_WORKER_ID", "worker"),
        ("IRIS_MULTIGPU_PROCESS_INDEX", "process_index"),
        ("IRIS_NODE_NAME", "node_name"),
    ):
        if value := os.environ.get(environment_name):
            resources[resource_name] = value
    return resources


def _ray_resources() -> dict[str, str]:
    if not ray.is_initialized():
        return {}
    try:
        context = ray.get_runtime_context()
    except Exception:
        logger.warning("Could not read Ray identity for telemetry; continuing without it", exc_info=True)
        return {}

    resources: dict[str, str] = {}
    for name, getter_name, keep_zero in (
        ("ray_job_id", "get_job_id", False),
        ("ray_task_id", "get_task_id", False),
        ("ray_task_attempt", "get_task_attempt_number", True),
        ("actor_uid", "get_actor_id", False),
        ("node_uid", "get_node_id", False),
    ):
        try:
            raw_value = getattr(context, getter_name)()
        except Exception:
            logger.warning(f"Could not read {name} for telemetry; continuing without it", exc_info=True)
            continue
        if raw_value is None:
            continue
        value = str(raw_value)
        if value and (keep_zero or set(value) != {"0"}):
            resources[name] = value
    return resources


def _resources(config: TelemetryConfig, role: str) -> dict[str, str]:
    resources = {
        **_iris_resources(),
        **_ray_resources(),
        "host": socket.gethostname(),
        "run_id": config.run_id or "",
        "execution_uid": config.execution_uid or "",
        "role": role,
    }
    if config.serving_job_id:
        resources["serving_job_id"] = config.serving_job_id
    if config.training_loop is not None:
        resources["training_loop"] = config.training_loop.value
    return resources


def phase_attributes(*, phase: str, root: str, parent: str | None, clock_domain: str) -> dict[str, str]:
    """Attributes that place one duration in a phase tree; a root carries no parent."""
    attributes = {"phase": phase, "root": root, "clock_domain": clock_domain}
    if parent is not None:
        attributes["parent"] = parent
    return attributes


@contextlib.contextmanager
def critical_phase(phase: Literal["rollout_or_inference_wait", "train_step"], step: int) -> Iterator[None]:
    started = time.perf_counter()
    outcome = "success"
    try:
        yield
    except BaseException:
        outcome = "failure"
        raise
    finally:
        phase_duration.record(
            time.perf_counter() - started,
            attributes={
                "phase": phase,
                "clock_domain": "critical_path",
                "role": TRAINER_ROLE,
                "outcome": outcome,
                "step": str(step),
            },
        )


def record_policy_step(step: int) -> None:
    progress_time = time.time()
    _process_state.policy_step = step
    _process_state.last_progress_timestamp = progress_time
    attributes = {"work_kind": "policy_step", "role": TRAINER_ROLE}
    work_completed.add(1, attributes={**attributes, "step": str(step)})
    progress_timestamp.set(progress_time, attributes=attributes)
    policy_step.set(step, attributes={"role": TRAINER_ROLE})


def record_generated_work(
    response_ids: Sequence[Sequence[int]], is_last_step: Sequence[bool] | None, weights_step: int
) -> None:
    """Count generated work against the policy version that produced it.

    Not against the step that recorded it: the producer runs before the group is enqueued, and a
    group can wait in the buffer across step boundaries, so the producer cannot know which step will
    consume it. Staleness is recorded by `record_rollout_staleness` where the trainer measures it.
    """
    sample_count = len(response_ids)
    rollout_count = sample_count if is_last_step is None else sum(is_last_step)
    generated_token_count = sum(len(response) for response in response_ids)
    progress_time = time.time()
    if sample_count:
        _process_state.last_progress_timestamp = progress_time
    for work_kind, count in (
        ("rollout", rollout_count),
        ("sample", sample_count),
        ("generated_token", generated_token_count),
    ):
        if count:
            work_completed.add(
                count,
                attributes={"work_kind": work_kind, "role": TRAINER_ROLE, "weights_step": str(weights_step)},
            )
    if rollout_count:
        progress_timestamp.set(
            progress_time,
            attributes={"work_kind": "rollout", "role": TRAINER_ROLE},
        )


def record_rollout_staleness(stalenesses: Sequence[int], step: int) -> None:
    """How far behind the consuming step each admitted group's policy was.

    Measured where the trainer measures it, which is the only place it is known: the same values it
    asserts against `max_staleness_steps` and reports as `async/staleness_*`.
    """
    for staleness in stalenesses:
        rollout_staleness.record(staleness, attributes={"role": TRAINER_ROLE, "step": str(step)})


def record_rollout_buffer(depth: int, queue_capacity: int) -> None:
    _process_state.queue_depth = depth
    _process_state.queue_capacity = queue_capacity
    attributes = {"queue": "rollout_buffer", "role": TRAINER_ROLE}
    rollout_queue_depth.set(depth, attributes=attributes)
    rollout_capacity.set(queue_capacity, attributes=attributes)


def _process_memory_snapshot() -> dict[str, int]:
    """Return process and host memory in bytes, including USS/PSS where supported."""
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        host = psutil.virtual_memory()
        values = {
            "rss": int(memory.rss),
            "vms": int(memory.vms),
            "system_used": int(host.used),
            "system_available": int(host.available),
        }
        try:
            full_memory = process.memory_full_info()
        except (psutil.AccessDenied, psutil.NoSuchProcess, AttributeError):
            full_memory = None
        if full_memory is not None:
            for name in ("uss", "pss"):
                value = getattr(full_memory, name, None)
                if value is not None:
                    values[name] = int(value)
        return values
    except (ImportError, OSError):
        return {}


def record_generation_retention_snapshot(
    *,
    groups: Mapping[str, int],
    rows: Mapping[str, int],
    estimated_bytes: Mapping[str, int],
    field_estimated_bytes: Mapping[tuple[str, str], int],
    producers: Mapping[str, int],
    settled_groups: int,
    settled_rows: int,
    boundary: str | None,
) -> None:
    """Publish a low-cardinality snapshot of driver-owned rollout memory."""

    for owner in _RETENTION_OWNERS:
        attributes = {"owner": owner}
        generation_retained_groups.set(groups.get(owner, 0), attributes=attributes)
        generation_retained_rows.set(rows.get(owner, 0), attributes=attributes)
        generation_retained_estimated_bytes.set(estimated_bytes.get(owner, 0), attributes=attributes)

        selected_field_bytes: Counter[str] = Counter()
        for (field_owner, field), value in field_estimated_bytes.items():
            if field_owner == owner:
                selected_field_bytes[field if field in _RETENTION_FIELDS else "other"] += value
        for field in _RETENTION_FIELDS:
            generation_retained_estimated_bytes.set(
                selected_field_bytes.get(field, 0),
                attributes={"owner": owner, "field": field},
            )

    for state in _PRODUCER_STATES:
        generation_producers.set(producers.get(state, 0), attributes={"state": state})
    generation_retention_events.set(settled_groups, attributes={"kind": "settled_group"})
    generation_retention_events.set(settled_rows, attributes={"kind": "settled_row"})
    for kind, value in _process_memory_snapshot().items():
        process_memory_bytes.set(value, attributes={"kind": kind})
    if boundary is not None:
        generation_memory_boundaries.add(1, attributes={"boundary": boundary})
    record_telemetry_health()


def generation_route(env_extras: Sequence[Mapping[str, object]] | None) -> str:
    """Classify a generation group without placing row identities in metric labels."""
    routes: set[str] = set()
    for extras in env_extras or ():
        extra_info = extras.get("extra_info")
        ultra = extra_info.get("nemotron_ultra") if isinstance(extra_info, Mapping) else None
        route = ultra.get("route") if isinstance(ultra, Mapping) else None
        routes.add("harbor" if route == "terminal_bench" else "gym")
    if not routes:
        return "unknown"
    return next(iter(routes)) if len(routes) == 1 else "mixed"


def record_generation_input_coverage(env_extras: Sequence[Mapping[str, object]] | None) -> int:
    """Count logical input coverage before admission can discard a completed group."""
    coverage: set[tuple[str, str, str]] = set()
    for extras in env_extras or ():
        extra_info = extras.get("extra_info")
        ultra = extra_info.get("nemotron_ultra") if isinstance(extra_info, Mapping) else None
        if not isinstance(ultra, Mapping):
            continue
        blend = ultra.get("blend")
        agent = ultra.get("agent")
        if not isinstance(blend, str) or not blend or not isinstance(agent, str) or not agent:
            continue
        route = "harbor" if ultra.get("route") == "terminal_bench" else "gym"
        coverage.add((blend, agent, route))
    for blend, agent, route in coverage:
        generation_input_coverage.add(1, attributes={"blend": blend, "agent": agent, "route": route})
    if coverage:
        record_telemetry_health()
    return len(coverage)


@contextlib.contextmanager
def generation_group(route: str) -> Iterator[None]:
    started = time.perf_counter()
    outcome = "success"
    _generation_active[route] += 1
    generation_active_groups.set(_generation_active[route], attributes={"route": route})
    try:
        yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException:
        outcome = "failure"
        raise
    finally:
        _generation_active[route] -= 1
        generation_active_groups.set(_generation_active[route], attributes={"route": route})
        attributes = {"route": route, "outcome": outcome}
        generation_groups.add(1, attributes=attributes)
        generation_group_duration.record(time.perf_counter() - started, attributes=attributes)


def record_admission(
    *,
    inspected: int,
    admitted: int,
    retried: int,
    discarded: int,
    reasons: Mapping[str, int],
) -> None:
    for outcome, count in (
        ("inspected", inspected),
        ("admitted", admitted),
        ("retried", retried),
        ("discarded", discarded),
    ):
        if count:
            admission_groups.add(count, attributes={"outcome": outcome, "reason": "none"})
    for reason, count in reasons.items():
        if count:
            admission_groups.add(count, attributes={"outcome": "rejected", "reason": str(reason)})


def _publish_executor_counts(work_kind: str) -> None:
    counts = _executor_counts[work_kind]
    for state in ("queued", "active"):
        executor_work.set(counts[state], attributes={"work_kind": work_kind, "state": state})


async def run_in_executor_observed(
    executor,
    work_kind: str,
    func: Callable,
    /,
    *args,
    **kwargs,
):
    """Run synchronous work while separating pool queue delay from execution time."""
    submitted = time.perf_counter()
    with _executor_lock:
        _executor_counts[work_kind]["queued"] += 1
        _publish_executor_counts(work_kind)

    def invoke():
        started = time.perf_counter()
        with _executor_lock:
            _executor_counts[work_kind]["queued"] -= 1
            _executor_counts[work_kind]["active"] += 1
            _publish_executor_counts(work_kind)
        executor_queue_delay.record(started - submitted, attributes={"work_kind": work_kind})
        outcome = "success"
        try:
            return func(*args, **kwargs)
        except BaseException:
            outcome = "failure"
            raise
        finally:
            executor_duration.record(
                time.perf_counter() - started,
                attributes={"work_kind": work_kind, "outcome": outcome},
            )
            with _executor_lock:
                _executor_counts[work_kind]["active"] -= 1
                _publish_executor_counts(work_kind)

    future = asyncio.get_running_loop().run_in_executor(executor, functools.partial(invoke))
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        executor_cancellations.add(1, attributes={"work_kind": work_kind})
        raise


class JudgeTelemetryObserver:
    """Duck-typed observer injected into the dependency-light skyrl-gym judge."""

    def request(self, *, duration_seconds: float, status: str, attempt: int, will_retry: bool) -> None:
        attributes = {
            "status": status,
            "attempt": str(attempt),
            "will_retry": str(will_retry).lower(),
            "subsystem": "judge",
        }
        judge_request_duration.record(duration_seconds, attributes=attributes)
        judge_requests.add(1, attributes=attributes)

    def parse(self, *, outcome: str, attempt: int) -> None:
        judge_parse_events.add(
            1,
            attributes={"outcome": outcome, "attempt": str(attempt), "subsystem": "judge"},
        )

    def cohort(self, *, duration_seconds: float, outcome: str) -> None:
        judge_cohort_duration.record(
            duration_seconds,
            attributes={"outcome": outcome, "subsystem": "judge"},
        )

    def executor(self, *, queue_seconds: float, duration_seconds: float, outcome: str) -> None:
        attributes = {"work_kind": "judge", "subsystem": "judge"}
        executor_queue_delay.record(queue_seconds, attributes=attributes)
        executor_duration.record(duration_seconds, attributes={**attributes, "outcome": outcome})


def record_telemetry_health() -> None:
    record = getattr(telemetry, "record_runtime_health", None)
    if record is not None:
        record()


class ProcessTelemetry:
    def __init__(self, config: TelemetryConfig, role: str) -> None:
        self._config = config
        self._role = role
        self._configured = False

    def __enter__(self) -> "ProcessTelemetry":
        if not _process_state.claim(self):
            logger.warning("Telemetry already has a process owner; leaving the nested lifecycle inert")
            return self
        if self._config.endpoint is None:
            return self
        if self._config.run_id is None or self._config.execution_uid is None:
            logger.warning("Telemetry requires run_id and execution_uid; export remains inert")
            return self
        telemetry.configure(
            endpoint=self._config.endpoint,
            service=SERVICE,
            attributes=_resources(self._config, self._role),
        )
        self._configured = telemetry.runtime_status().configured
        if self._configured:
            record_event("lifecycle", {"state": "started"}, attributes={"role": self._role})
            telemetry_smoke.set(1, attributes={"role": self._role, "state": "started"})
            record_telemetry_health()
        return self

    def collector_or_inert(self, collector: _BackgroundCollector) -> _BackgroundCollector:
        return collector if self._configured else _inert_collector

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        if self._configured:
            record_telemetry_health()
            export = telemetry.runtime_status()
            record_event(
                "terminal",
                {
                    "status": "completed" if exc_type is None else "failed",
                    "reason": "normal_exit" if exc_type is None else getattr(exc_type, "__name__", "exception"),
                    "export_queued_records": export.queued_records,
                    "export_lost_records": export.lost_records,
                    "policy_step": _process_state.policy_step,
                    "last_progress_time_seconds": _process_state.last_progress_timestamp,
                    "queue_depth": _process_state.queue_depth,
                    "queue_capacity": _process_state.queue_capacity,
                },
                attributes={"role": self._role},
            )
            telemetry.shutdown(SHUTDOWN_TIMEOUT_SECONDS)
        self._configured = False
        _process_state.release(self)
        return False


def process_telemetry(role: str) -> ProcessTelemetry:
    return ProcessTelemetry(TelemetryConfig.from_environment(), role)
