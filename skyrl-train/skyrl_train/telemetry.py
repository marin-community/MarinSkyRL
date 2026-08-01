import contextlib
import os
import socket
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from omegaconf import DictConfig

try:
    from rigging import telemetry as _rigging
except ImportError:
    _rigging = None


SERVICE = "marinskyrl"
ProcessRole = Literal["driver", "trainer"]

_active_process: "ProcessTelemetry | None" = None


def _instrument(kind: str, name: str, unit: str):
    if _rigging is None:
        return None
    try:
        return getattr(_rigging, kind)(name, unit=unit)
    except Exception:
        return None


_WORK_COMPLETED = _instrument("counter", "work_completed", "{item}")
_PHASE_DURATION_SECONDS = _instrument("histogram", "phase_duration_seconds", "s")
_PROGRESS_TIME_SECONDS = _instrument("gauge", "progress_time_seconds", "s")
_POLICY_STEP = _instrument("gauge", "policy_step", "{step}")
_QUEUE_DEPTH = _instrument("gauge", "queue_depth", "{item}")
_CAPACITY = _instrument("gauge", "capacity", "{item}")


@dataclass(frozen=True)
class TelemetryConfig:
    endpoint: str | None = None
    root_run_uid: str | None = None
    execution_uid: str | None = None
    serving_job_id: str | None = None
    shutdown_timeout_seconds: float = 2.0

    @classmethod
    def from_config(cls, cfg: DictConfig | Mapping[str, Any] | None) -> "TelemetryConfig":
        telemetry_cfg = cfg.get("telemetry", {}) if cfg is not None and hasattr(cfg, "get") else {}

        def text(key: str, environment_name: str) -> str | None:
            value = telemetry_cfg.get(key) if hasattr(telemetry_cfg, "get") else None
            if value is None:
                value = os.environ.get(environment_name)
            value = str(value).strip() if value is not None else ""
            return value or None

        try:
            shutdown_timeout = float(telemetry_cfg.get("shutdown_timeout_seconds", 2.0))
        except (TypeError, ValueError):
            shutdown_timeout = 2.0
        if not 0 <= shutdown_timeout <= 5:
            shutdown_timeout = 2.0
        return cls(
            endpoint=text("endpoint", "SKYRL_TELEMETRY_ENDPOINT"),
            root_run_uid=text("root_run_uid", "SKYRL_ROOT_RUN_UID"),
            execution_uid=text("execution_uid", "SKYRL_EXECUTION_UID"),
            serving_job_id=text("serving_job_id", "SKYRL_SERVING_JOB_ID"),
            shutdown_timeout_seconds=shutdown_timeout,
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
    try:
        import ray

        if not ray.is_initialized():
            return {}
        context = ray.get_runtime_context()
    except Exception:
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
                        "role": "trainer",
                        "outcome": outcome,
                    },
                )
        except Exception:
            pass


def record_policy_step(policy_step: int) -> None:
    if _active_process is not None:
        _active_process.policy_step = policy_step
        _active_process.last_progress_time_seconds = time.time()
    attributes = {"work_kind": "policy_step", "role": "trainer"}
    try:
        if _WORK_COMPLETED is not None:
            _WORK_COMPLETED.add(1, attributes=attributes)
        if _PROGRESS_TIME_SECONDS is not None:
            _PROGRESS_TIME_SECONDS.set(time.time(), attributes=attributes)
        if _POLICY_STEP is not None:
            _POLICY_STEP.set(policy_step, attributes={"role": "trainer"})
    except Exception:
        pass


def record_rollout_buffer(depth: int, capacity: int) -> None:
    if _active_process is not None:
        _active_process.queue_depth = depth
        _active_process.queue_capacity = capacity
    attributes = {"queue": "rollout_buffer", "role": "trainer"}
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

        _active_process = self
        if _rigging is None or self.config.endpoint is None:
            return self
        if self.config.root_run_uid is None or self.config.execution_uid is None:
            try:
                logger.warning("Telemetry requires root_run_uid and execution_uid; export remains inert")
            except Exception:
                pass
            return self
        try:
            _rigging.configure(
                endpoint=self.config.endpoint,
                service=SERVICE,
                attributes=_resources(self.config, self.role),
            )
            self.configured = bool(_rigging.runtime_status().configured)
        except Exception:
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


def process_telemetry(cfg: DictConfig | Mapping[str, Any] | None, role: ProcessRole) -> ProcessTelemetry:
    return ProcessTelemetry(TelemetryConfig.from_config(cfg), role)
