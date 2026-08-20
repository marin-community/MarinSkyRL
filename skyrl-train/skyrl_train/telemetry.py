import contextlib
import os
import socket
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import ray
from loguru import logger

try:
    from rigging import telemetry
except ImportError as error:
    # An installed rigging without the telemetry submodule raises ImportError, not
    # ModuleNotFoundError; the name check still keeps a failure inside rigging visible.
    if error.name != "rigging":
        raise
    from skyrl_train import inert_telemetry as telemetry


SERVICE = "marinskyrl"
DRIVER_ROLE = "driver"
TRAINER_ROLE = "trainer"
CONTROLLER_ROLE = "controller"
INFERENCE_ROLE = "inference"
SHUTDOWN_TIMEOUT_SECONDS = 2.0

# num_requests_running, num_requests_waiting, gpu_cache_usage_perc and num_preemptions_total are
# vLLM's own Prometheus spellings, so they read the same for the rollout engines and for marin's
# serving path. This repo scales gpu_cache_usage_perc to 0-100 while vLLM's gauge is 0-1, and
# marin's reader folds both spellings into one family. prefix_cache_hit_rate is a percentage here
# too. The rest carry reductions vLLM has no scalar name for. A reduction is an attribute rather than a name prefix, so one series carries peak and
# median.
ENGINE_SCOPE_ALL = "all"

work_completed = telemetry.counter("work_completed", unit="{item}")
phase_duration = telemetry.histogram("phase_duration_seconds", unit="s")
progress_timestamp = telemetry.gauge("progress_time_seconds", unit="s")
policy_step = telemetry.gauge("policy_step", unit="{step}")
rollout_queue_depth = telemetry.gauge("queue_depth", unit="{item}")
rollout_capacity = telemetry.gauge("capacity", unit="{item}")
requests_running = telemetry.gauge("num_requests_running", unit="{request}")
requests_waiting = telemetry.gauge("num_requests_waiting", unit="{request}")
gpu_cache_usage = telemetry.gauge("gpu_cache_usage_perc", unit="1")
prefix_cache_hit_rate = telemetry.gauge("prefix_cache_hit_rate", unit="%")
prompt_throughput = telemetry.gauge("prompt_throughput_tokens_per_second", unit="{token}/s")
generation_throughput = telemetry.gauge("generation_throughput_tokens_per_second", unit="{token}/s")
request_latency = telemetry.gauge("request_latency_seconds", unit="s")
requests_finished = telemetry.counter("requests_finished", unit="{request}")
requests_preempted = telemetry.counter("num_preemptions_total", unit="{request}")


class _Gauge(Protocol):
    def set(self, value: float, *, attributes: dict[str, str] | None = None) -> None: ...


class _Counter(Protocol):
    def add(self, value: float = 1.0, *, attributes: dict[str, str] | None = None) -> None: ...


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
        phase_duration.record(
            time.perf_counter() - started,
            attributes={
                "phase": phase,
                "clock_domain": "critical_path",
                "role": TRAINER_ROLE,
                "outcome": outcome,
            },
        )


def record_policy_step(step: int) -> None:
    progress_time = time.time()
    _process_state.policy_step = step
    _process_state.last_progress_timestamp = progress_time
    attributes = {"work_kind": "policy_step", "role": TRAINER_ROLE}
    work_completed.add(1, attributes=attributes)
    progress_timestamp.set(progress_time, attributes=attributes)
    policy_step.set(step, attributes={"role": TRAINER_ROLE})


def record_generated_work(response_ids: Sequence[Sequence[int]], is_last_step: Sequence[bool] | None) -> None:
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
            work_completed.add(count, attributes={"work_kind": work_kind, "role": TRAINER_ROLE})
    if rollout_count:
        progress_timestamp.set(
            progress_time,
            attributes={"work_kind": "rollout", "role": TRAINER_ROLE},
        )


# One engine payload's reduced values, as (instrument, statistic, key) triples. The aggregate and the
# per-engine payloads use different key names for the same quantity, so each scope brings its own map.
_AGGREGATE_GAUGES = (
    (requests_running, "peak", "total_peak_running_reqs"),
    (requests_running, "median", "avg_median_running_reqs"),
    (requests_waiting, "peak", "total_peak_waiting_reqs"),
    (requests_waiting, "median", "avg_median_waiting_reqs"),
    (gpu_cache_usage, "peak", "avg_peak_gpu_cache_usage_perc"),
    (gpu_cache_usage, "median", "avg_median_gpu_cache_usage_perc"),
    (prefix_cache_hit_rate, "peak", "avg_peak_prefix_cache_hit_rate"),
    (prefix_cache_hit_rate, "median", "avg_median_prefix_cache_hit_rate"),
    (prompt_throughput, "peak", "avg_peak_prompt_throughput"),
    (prompt_throughput, "median", "avg_median_prompt_throughput"),
    (generation_throughput, "peak", "avg_peak_generation_throughput"),
    (generation_throughput, "median", "avg_median_generation_throughput"),
)

_ENGINE_GAUGES = (
    (requests_running, "peak", "peak_running_reqs"),
    (requests_running, "median", "median_running_reqs"),
    (requests_waiting, "peak", "peak_waiting_reqs"),
    (requests_waiting, "median", "median_waiting_reqs"),
    (gpu_cache_usage, "peak", "peak_gpu_cache_usage_perc"),
    (gpu_cache_usage, "median", "median_gpu_cache_usage_perc"),
    (prefix_cache_hit_rate, "peak", "peak_prefix_cache_hit_rate"),
    (prefix_cache_hit_rate, "median", "median_prefix_cache_hit_rate"),
    (prompt_throughput, "peak", "peak_prompt_throughput"),
    (prompt_throughput, "median", "median_prompt_throughput"),
    (generation_throughput, "peak", "peak_generation_throughput"),
    (generation_throughput, "median", "median_generation_throughput"),
)

_LATENCY_STAGES = ("prefill", "decode", "e2e", "queued", "ttft")


def _record_engine_scope(
    payload: dict,
    scope: str,
    gauges: Sequence[tuple[_Gauge, str, str]],
    latency_key: Callable[[str, str], str],
    counters: Sequence[tuple[_Counter, str]],
) -> None:
    attributes = {"role": INFERENCE_ROLE, "engine": scope}
    for instrument, statistic, key in gauges:
        value = payload.get(key)
        if value is not None:
            instrument.set(float(value), attributes={**attributes, "statistic": statistic})
    for stage in _LATENCY_STAGES:
        for statistic in ("mean", "p90"):
            value = payload.get(latency_key(stage, statistic))
            if value is not None:
                request_latency.set(float(value), attributes={**attributes, "stage": stage, "statistic": statistic})
    for instrument, key in counters:
        count = payload.get(key)
        if count is not None:
            instrument.add(float(count), attributes=attributes)


def record_engine_stats(stats: dict) -> None:
    """Export one step's inference-engine reductions.

    The aggregate is always exported. Every engine is also exported individually once there is more
    than one, because with a single engine the per-engine series would repeat the aggregate.

    ``get_stats`` resets each engine's accumulators as it reads them, so every value here describes
    the step just finished: the gauges are that step's peak and median, and the counts are that
    step's totals, which is why they are added as counter deltas rather than set.
    """
    if not stats.get("num_engines"):
        return
    _record_engine_scope(
        stats,
        ENGINE_SCOPE_ALL,
        _AGGREGATE_GAUGES,
        lambda stage, statistic: f"{'max' if statistic == 'p90' else 'avg'}_latency_{stage}_{statistic}",
        ((requests_finished, "total_finished_requests"), (requests_preempted, "total_preempted_reqs")),
    )
    if stats["num_engines"] <= 1:
        return
    for index, engine in enumerate(stats.get("engines", ())):
        _record_engine_scope(
            engine,
            str(index),
            _ENGINE_GAUGES,
            lambda stage, statistic: f"latency_{stage}_{statistic}",
            ((requests_finished, "latency_num_finished_requests"), (requests_preempted, "total_preempted_reqs")),
        )


def record_rollout_buffer(depth: int, queue_capacity: int) -> None:
    _process_state.queue_depth = depth
    _process_state.queue_capacity = queue_capacity
    attributes = {"queue": "rollout_buffer", "role": TRAINER_ROLE}
    rollout_queue_depth.set(depth, attributes=attributes)
    rollout_capacity.set(queue_capacity, attributes=attributes)


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
        if self._config.root_run_uid is None or self._config.execution_uid is None:
            logger.warning("Telemetry requires root_run_uid and execution_uid; export remains inert")
            return self
        telemetry.configure(
            endpoint=self._config.endpoint,
            service=SERVICE,
            attributes=_resources(self._config, self._role),
        )
        self._configured = telemetry.runtime_status().configured
        if self._configured:
            telemetry.event("lifecycle", {"state": "started"}, attributes={"role": self._role})
        return self

    def collector_or_inert(self, collector: _BackgroundCollector) -> _BackgroundCollector:
        return collector if self._configured else _inert_collector

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        if self._configured:
            export = telemetry.runtime_status()
            telemetry.event(
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
