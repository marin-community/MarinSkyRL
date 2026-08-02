import contextlib
import os
import socket
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import ray
from loguru import logger

try:
    from rigging import telemetry
except ImportError:
    from skyrl_train import inert_telemetry as telemetry


SERVICE = "marinskyrl"
DRIVER_ROLE = "driver"
TRAINER_ROLE = "trainer"
CONTROLLER_ROLE = "controller"
SHUTDOWN_TIMEOUT_SECONDS = 2.0

work_completed = telemetry.counter("work_completed", unit="{item}")
phase_duration_seconds = telemetry.histogram("phase_duration_seconds", unit="s")
progress_time_seconds = telemetry.gauge("progress_time_seconds", unit="s")
policy_step = telemetry.gauge("policy_step", unit="{step}")
queue_depth = telemetry.gauge("queue_depth", unit="{item}")
capacity = telemetry.gauge("capacity", unit="{item}")


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
    policy_step: int | None = None
    last_progress_time_seconds: float | None = None
    queue_depth: int | None = None
    queue_capacity: int | None = None


_inactive_state = _ProcessState()
_active_state = _inactive_state
_active_process: "ProcessTelemetry | None" = None


@dataclass(frozen=True)
class TelemetryConfig:
    endpoint: str | None = None
    root_run_uid: str | None = None
    execution_uid: str | None = None
    serving_job_id: str | None = None

    @classmethod
    def from_environment(cls) -> "TelemetryConfig":
        def text(environment_name: str) -> str | None:
            value = os.environ.get(environment_name, "").strip()
            return value or None

        return cls(
            endpoint=text("SKYRL_TELEMETRY_ENDPOINT"),
            root_run_uid=text("SKYRL_ROOT_RUN_UID"),
            execution_uid=text("SKYRL_EXECUTION_UID"),
            serving_job_id=text("SKYRL_SERVING_JOB_ID"),
        )


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
        logger.warning("Could not read Ray identity for telemetry; continuing without it")
        return {}

    try:
        values = (
            ("ray_job_id", context.get_job_id(), False),
            ("ray_task_id", context.get_task_id(), False),
            ("ray_task_attempt", context.get_task_attempt_number(), True),
            ("actor_uid", context.get_actor_id(), False),
            ("node_uid", context.get_node_id(), False),
        )
    except Exception:
        logger.warning("Could not read Ray identity for telemetry; continuing without it", exc_info=True)
        return {}

    resources: dict[str, str] = {}
    for name, raw_value, keep_zero in values:
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
        "root_run_uid": config.root_run_uid or "",
        "execution_uid": config.execution_uid or "",
        "role": role,
    }
    if config.serving_job_id:
        resources["serving_job_id"] = config.serving_job_id
    return resources


@contextlib.contextmanager
def critical_phase(phase: Literal["rollout_or_inference_wait", "train_step"]) -> Iterator[None]:
    started = time.perf_counter()
    outcome = "success"
    try:
        yield
    except BaseException:
        outcome = "failure"
        raise
    finally:
        try:
            phase_duration_seconds.record(
                time.perf_counter() - started,
                attributes={
                    "phase": phase,
                    "clock_domain": "critical_path",
                    "role": TRAINER_ROLE,
                    "outcome": outcome,
                },
            )
        except Exception:
            pass


def record_policy_step(step: int) -> None:
    progress_time = time.time()
    _active_state.policy_step = step
    _active_state.last_progress_time_seconds = progress_time
    attributes = {"work_kind": "policy_step", "role": TRAINER_ROLE}
    try:
        work_completed.add(1, attributes=attributes)
        progress_time_seconds.set(progress_time, attributes=attributes)
        policy_step.set(step, attributes={"role": TRAINER_ROLE})
    except Exception:
        pass


def record_generated_work(response_ids: Sequence[Sequence[int]], is_last_step: Sequence[bool] | None) -> None:
    sample_count = len(response_ids)
    rollout_count = sample_count if is_last_step is None else sum(is_last_step)
    generated_token_count = sum(len(response) for response in response_ids)
    progress_time = time.time()
    if sample_count:
        _active_state.last_progress_time_seconds = progress_time
    try:
        for work_kind, count in (
            ("rollout", rollout_count),
            ("sample", sample_count),
            ("generated_token", generated_token_count),
        ):
            if count:
                work_completed.add(count, attributes={"work_kind": work_kind, "role": TRAINER_ROLE})
        if rollout_count:
            progress_time_seconds.set(
                progress_time,
                attributes={"work_kind": "rollout", "role": TRAINER_ROLE},
            )
    except Exception:
        pass


def record_rollout_buffer(depth: int, capacity: int) -> None:
    _active_state.queue_depth = depth
    _active_state.queue_capacity = capacity
    attributes = {"queue": "rollout_buffer", "role": TRAINER_ROLE}
    try:
        queue_depth.set(depth, attributes=attributes)
        capacity.set(capacity, attributes=attributes)
    except Exception:
        pass


class ProcessTelemetry:
    def __init__(self, config: TelemetryConfig, role: str) -> None:
        self._config = config
        self._role = role
        self._configured = False
        self._closed = False
        self._state = _ProcessState()

    def __enter__(self) -> "ProcessTelemetry":
        global _active_process, _active_state

        if _active_process is not None:
            logger.warning("Telemetry already has a process owner; leaving the nested lifecycle inert")
            return self
        _active_process = self
        _active_state = self._state
        if self._config.endpoint is None:
            return self
        if self._config.root_run_uid is None or self._config.execution_uid is None:
            logger.warning("Telemetry requires root_run_uid and execution_uid; export remains inert")
            return self
        try:
            telemetry.configure(
                endpoint=self._config.endpoint,
                service=SERVICE,
                attributes=_resources(self._config, self._role),
            )
            self._configured = bool(telemetry.runtime_status().configured)
        except Exception:
            logger.warning("Could not configure telemetry; export remains inert", exc_info=True)
            self._configured = False
        if self._configured:
            try:
                telemetry.event("lifecycle", {"state": "started"}, attributes={"role": self._role})
            except Exception:
                logger.warning("Could not export telemetry lifecycle start", exc_info=True)
        return self

    def background_collector(self, collector: _BackgroundCollector) -> _BackgroundCollector:
        return collector if self._configured else _inert_collector

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        self.close(
            status="completed" if exc_type is None else "failed",
            reason="normal_exit" if exc_type is None else getattr(exc_type, "__name__", "exception"),
        )
        return False

    def close(self, *, status: str, reason: str) -> None:
        global _active_process, _active_state

        if self._closed:
            return
        self._closed = True
        if self._configured:
            try:
                export = telemetry.runtime_status()
                telemetry.event(
                    "terminal",
                    {
                        "status": status,
                        "reason": reason[:512],
                        "export_queued_records": export.queued_records,
                        "export_lost_records": export.lost_records,
                        "policy_step": self._state.policy_step,
                        "last_progress_time_seconds": self._state.last_progress_time_seconds,
                        "queue_depth": self._state.queue_depth,
                        "queue_capacity": self._state.queue_capacity,
                    },
                    attributes={"role": self._role},
                )
            except Exception:
                logger.warning("Could not export telemetry terminal state", exc_info=True)
            self._drain_and_shutdown()
        self._configured = False
        if _active_process is self:
            _active_process = None
            _active_state = _inactive_state

    def _drain_and_shutdown(self) -> None:
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
        try:
            while telemetry.runtime_status().queued_records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))
        except Exception:
            logger.warning("Could not read telemetry queue status during shutdown", exc_info=True)
        try:
            telemetry.shutdown(max(0.0, deadline - time.monotonic()))
        except Exception:
            logger.warning("Could not shut down telemetry exporter", exc_info=True)


def process_telemetry(role: str) -> ProcessTelemetry:
    return ProcessTelemetry(TelemetryConfig.from_environment(), role)
