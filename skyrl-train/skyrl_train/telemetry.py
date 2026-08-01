import contextlib
import os
import socket
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import ray
from loguru import logger

try:
    from rigging import telemetry as _rigging
except ImportError:
    _rigging = None


SERVICE = "marinskyrl"
ProcessRole = Literal["driver", "trainer"]
TRAINER_ROLE: ProcessRole = "trainer"
SHUTDOWN_TIMEOUT_SECONDS = 2.0

_active_process: "ProcessTelemetry | None" = None

if _rigging is None:
    _WORK_COMPLETED = _PHASE_DURATION_SECONDS = _PROGRESS_TIME_SECONDS = None
    _POLICY_STEP = _QUEUE_DEPTH = _CAPACITY = None
else:
    _WORK_COMPLETED = _rigging.counter("work_completed", unit="{item}")
    _PHASE_DURATION_SECONDS = _rigging.histogram("phase_duration_seconds", unit="s")
    _PROGRESS_TIME_SECONDS = _rigging.gauge("progress_time_seconds", unit="s")
    _POLICY_STEP = _rigging.gauge("policy_step", unit="{step}")
    _QUEUE_DEPTH = _rigging.gauge("queue_depth", unit="{item}")
    _CAPACITY = _rigging.gauge("capacity", unit="{item}")


@dataclass(frozen=True)
class TelemetryConfig:
    endpoint: str | None = None
    root_run_uid: str | None = None
    execution_uid: str | None = None
    serving_job_id: str | None = None
    shutdown_timeout_seconds: float = SHUTDOWN_TIMEOUT_SECONDS

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

    resources: dict[str, str] = {}
    for name, getter, keep_zero in (
        ("ray_job_id", "get_job_id", False),
        ("ray_task_id", "get_task_id", False),
        ("ray_task_attempt", "get_task_attempt_number", True),
        ("actor_uid", "get_actor_id", False),
        ("node_uid", "get_node_id", False),
    ):
        try:
            raw_value = getattr(context, getter)()
        except Exception:
            continue
        if raw_value is None:
            continue
        value = str(raw_value)
        if value and (keep_zero or set(value) != {"0"}):
            resources[name] = value
    return resources


def _resources(config: TelemetryConfig, role: ProcessRole) -> dict[str, str]:
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
            if _PHASE_DURATION_SECONDS is not None:
                _PHASE_DURATION_SECONDS.record(
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


def record_policy_step(policy_step: int) -> None:
    progress_time = time.time()
    if _active_process is not None:
        _active_process.policy_step = policy_step
        _active_process.last_progress_time_seconds = progress_time
    attributes = {"work_kind": "policy_step", "role": TRAINER_ROLE}
    try:
        if _WORK_COMPLETED is not None:
            _WORK_COMPLETED.add(1, attributes=attributes)
        if _PROGRESS_TIME_SECONDS is not None:
            _PROGRESS_TIME_SECONDS.set(progress_time, attributes=attributes)
        if _POLICY_STEP is not None:
            _POLICY_STEP.set(policy_step, attributes={"role": TRAINER_ROLE})
    except Exception:
        pass


def record_rollout_buffer(depth: int, capacity: int) -> None:
    if _active_process is not None:
        _active_process.queue_depth = depth
        _active_process.queue_capacity = capacity
    attributes = {"queue": "rollout_buffer", "role": TRAINER_ROLE}
    try:
        if _QUEUE_DEPTH is not None:
            _QUEUE_DEPTH.set(depth, attributes=attributes)
        if _CAPACITY is not None:
            _CAPACITY.set(capacity, attributes=attributes)
    except Exception:
        pass


class ProcessTelemetry:
    def __init__(self, config: TelemetryConfig, role: ProcessRole) -> None:
        self.config = config
        self.role = role
        self.configured = False
        self.closed = False
        self.policy_step: int | None = None
        self.last_progress_time_seconds: float | None = None
        self.queue_depth: int | None = None
        self.queue_capacity: int | None = None

    def __enter__(self) -> "ProcessTelemetry":
        global _active_process

        if _active_process is not None:
            logger.warning("Telemetry already has a process owner; leaving the nested lifecycle inert")
            return self
        _active_process = self
        if _rigging is None or self.config.endpoint is None:
            return self
        if self.config.root_run_uid is None or self.config.execution_uid is None:
            logger.warning("Telemetry requires root_run_uid and execution_uid; export remains inert")
            return self
        try:
            _rigging.configure(
                endpoint=self.config.endpoint,
                service=SERVICE,
                attributes=_resources(self.config, self.role),
            )
            self.configured = bool(_rigging.runtime_status().configured)
        except Exception:
            logger.warning("Could not configure telemetry; export remains inert")
            self.configured = False
        if self.configured:
            try:
                _rigging.event("lifecycle", {"state": "started"}, attributes={"role": self.role})
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        self.close(
            status="completed" if exc_type is None else "failed",
            reason="normal_exit" if exc_type is None else getattr(exc_type, "__name__", "exception"),
        )
        return False

    def close(self, *, status: str, reason: str) -> None:
        global _active_process

        if self.closed:
            return
        self.closed = True
        if _rigging is not None and self.configured:
            try:
                export = _rigging.runtime_status()
                _rigging.event(
                    "terminal",
                    {
                        "status": status,
                        "reason": reason[:512],
                        "export_queued_records": export.queued_records,
                        "export_lost_records": export.lost_records,
                        "policy_step": self.policy_step,
                        "last_progress_time_seconds": self.last_progress_time_seconds,
                        "queue_depth": self.queue_depth,
                        "queue_capacity": self.queue_capacity,
                    },
                    attributes={"role": self.role},
                )
            except Exception:
                pass
            self._drain_and_shutdown()
        self.configured = False
        if _active_process is self:
            _active_process = None

    def _drain_and_shutdown(self) -> None:
        deadline = time.monotonic() + self.config.shutdown_timeout_seconds
        try:
            while _rigging.runtime_status().queued_records:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))
        except Exception:
            pass
        try:
            _rigging.shutdown(max(0.0, deadline - time.monotonic()))
        except Exception:
            pass


def process_telemetry(role: ProcessRole) -> ProcessTelemetry:
    return ProcessTelemetry(TelemetryConfig.from_environment(), role)
