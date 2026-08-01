import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train import fully_async_trainer, telemetry
from skyrl_train.fully_async_trainer import GeneratedOutputGroup, _put_rollout_with_backpressure


class RecordingInstrument:
    def __init__(self, kind: str, name: str, unit: str) -> None:
        self.kind = kind
        self.name = name
        self.unit = unit
        self.calls: list[tuple[str, float, dict[str, str]]] = []

    def add(self, value=1, *, attributes=None) -> None:
        self.calls.append(("add", value, attributes or {}))

    def set(self, value, *, attributes=None) -> None:
        self.calls.append(("set", value, attributes or {}))

    def record(self, value, *, attributes=None) -> None:
        self.calls.append(("record", value, attributes or {}))


class RecordingBackend:
    def __init__(self) -> None:
        self.configured = False
        self.configure_calls: list[dict] = []
        self.instruments: list[RecordingInstrument] = []
        self.events: list[tuple[str, dict, dict[str, str]]] = []
        self.shutdown_timeouts: list[float] = []

    def configure(self, **kwargs) -> None:
        self.configure_calls.append(kwargs)
        self.configured = True

    def runtime_status(self):
        return SimpleNamespace(configured=self.configured, queued_records=0, lost_records=2)

    def counter(self, name: str, *, unit: str = "") -> RecordingInstrument:
        return self._instrument("counter", name, unit)

    def gauge(self, name: str, *, unit: str = "") -> RecordingInstrument:
        return self._instrument("gauge", name, unit)

    def histogram(self, name: str, *, unit: str = "") -> RecordingInstrument:
        return self._instrument("histogram", name, unit)

    def _instrument(self, kind: str, name: str, unit: str) -> RecordingInstrument:
        instrument = RecordingInstrument(kind, name, unit)
        self.instruments.append(instrument)
        return instrument

    def event(self, name: str, body: dict, *, attributes=None) -> None:
        self.events.append((name, body, attributes or {}))

    def shutdown(self, timeout: float) -> None:
        self.shutdown_timeouts.append(timeout)
        self.configured = False


@pytest.fixture(autouse=True)
def recording_backend(monkeypatch: pytest.MonkeyPatch) -> RecordingBackend:
    backend = RecordingBackend()
    monkeypatch.setattr(telemetry, "_rigging", backend)
    for target, kind, name, unit in (
        ("_WORK_COMPLETED", "counter", "work_completed", "{item}"),
        ("_PHASE_DURATION_SECONDS", "histogram", "phase_duration_seconds", "s"),
        ("_PROGRESS_TIME_SECONDS", "gauge", "progress_time_seconds", "s"),
        ("_POLICY_STEP", "gauge", "policy_step", "{step}"),
        ("_QUEUE_DEPTH", "gauge", "queue_depth", "{item}"),
        ("_CAPACITY", "gauge", "capacity", "{item}"),
    ):
        monkeypatch.setattr(telemetry, target, getattr(backend, kind)(name, unit=unit))
    monkeypatch.setattr(telemetry, "_active_process", None)
    return backend


def configured() -> OmegaConf:
    return OmegaConf.create(
        {
            "telemetry": {
                "endpoint": "http://finelog.test/v1/telemetry",
                "root_run_uid": "root-1",
                "execution_uid": "execution-2",
                "serving_job_id": "serving-3",
                "service": "operator-override-is-ignored",
                "shutdown_timeout_seconds": 0.03,
            }
        }
    )


def test_trainer_lifecycle_uses_canonical_identity_and_signals(
    monkeypatch: pytest.MonkeyPatch, recording_backend: RecordingBackend
) -> None:
    monkeypatch.setenv("SKYRL_TELEMETRY_ENDPOINT", "http://finelog.test/v1/telemetry")
    monkeypatch.setenv("SKYRL_ROOT_RUN_UID", "root-1")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "execution-2")
    monkeypatch.setattr(telemetry, "_iris_resources", lambda: {"job_id": "/u/job", "attempt": "2"})
    monkeypatch.setattr(
        telemetry,
        "_ray_resources",
        lambda: {"ray_job_id": "ray-job", "actor_uid": "actor-1", "node_uid": "node-1"},
    )

    config = OmegaConf.create(
        {
            "telemetry": {
                "serving_job_id": "serving-3",
                "service": "operator-override-is-ignored",
                "shutdown_timeout_seconds": 0.03,
            }
        }
    )
    with telemetry.process_telemetry(config, "trainer"):
        telemetry.record_policy_step(7)
        telemetry.record_rollout_buffer(2, 4)

    call = recording_backend.configure_calls[0]
    assert call["service"] == "marinskyrl"
    assert {
        "root_run_uid": "root-1",
        "execution_uid": "execution-2",
        "serving_job_id": "serving-3",
        "role": "trainer",
        "job_id": "/u/job",
        "attempt": "2",
        "ray_job_id": "ray-job",
        "actor_uid": "actor-1",
        "node_uid": "node-1",
    }.items() <= call["attributes"].items()
    assert "run_id" not in call["attributes"]
    assert [event[0] for event in recording_backend.events] == ["lifecycle", "terminal"]
    assert {
        "status": "completed",
        "policy_step": 7,
        "queue_depth": 2,
        "queue_capacity": 4,
        "export_lost_records": 2,
    }.items() <= recording_backend.events[-1][1].items()
    assert len(recording_backend.shutdown_timeouts) == 1
    assert 0 < recording_backend.shutdown_timeouts[0] <= 0.03

    instruments = {(item.name, item.unit): item for item in recording_backend.instruments}
    work = instruments[("work_completed", "{item}")]
    assert work.calls == [("add", 1, {"work_kind": "policy_step", "role": "trainer"})]
    assert instruments[("phase_duration_seconds", "s")].calls == []


def test_phase_observes_system_exit_and_records_exclusive_clock(
    recording_backend: RecordingBackend,
) -> None:
    with pytest.raises(SystemExit):
        with telemetry.critical_phase("rollout_or_inference_wait"):
            raise SystemExit(9)

    instrument = next(item for item in recording_backend.instruments if item.name == "phase_duration_seconds")
    assert (instrument.name, instrument.unit) == ("phase_duration_seconds", "s")
    assert {
        "phase": "rollout_or_inference_wait",
        "clock_domain": "critical_path",
        "role": "trainer",
        "outcome": "failure",
    }.items() <= instrument.calls[0][2].items()


def test_unconfigured_export_is_inert(recording_backend: RecordingBackend) -> None:
    with telemetry.process_telemetry(OmegaConf.create({}), "driver"):
        pass

    assert recording_backend.configure_calls == []
    assert recording_backend.events == []
    assert recording_backend.shutdown_timeouts == []


class FailingInstrument(RecordingInstrument):
    def add(self, value=1, *, attributes=None) -> None:
        raise RuntimeError("export unavailable")

    set = add
    record = add


class FailingBackend(RecordingBackend):
    def _instrument(self, kind: str, name: str, unit: str) -> RecordingInstrument:
        return FailingInstrument(kind, name, unit)

    def event(self, name: str, body: dict, *, attributes=None) -> None:
        raise RuntimeError("export unavailable")

    def shutdown(self, timeout: float) -> None:
        self.shutdown_timeouts.append(timeout)
        raise RuntimeError("export unavailable")


def test_exporter_and_shutdown_failures_do_not_change_application_result(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FailingBackend()
    monkeypatch.setattr(telemetry, "_rigging", backend)
    monkeypatch.setattr(telemetry, "_WORK_COMPLETED", backend.counter("work_completed", unit="{item}"))
    monkeypatch.setattr(telemetry, "_PHASE_DURATION_SECONDS", backend.histogram("phase_duration_seconds", unit="s"))
    monkeypatch.setattr(telemetry, "_PROGRESS_TIME_SECONDS", backend.gauge("progress_time_seconds", unit="s"))

    with telemetry.process_telemetry(configured(), "trainer"):
        with telemetry.critical_phase("train_step"):
            telemetry.record_policy_step(1)
            result = 17

    assert result == 17
    assert telemetry._active_process is None
    assert len(backend.shutdown_timeouts) == 1


def test_shutdown_drains_multiple_event_records_within_one_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class DrainingBackend(RecordingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.pending: list[str] = []
            self.delivered: list[str] = []
            self.drain_started = False

        def runtime_status(self):
            if self.drain_started and self.pending:
                self.delivered.append(self.pending.pop(0))
            return SimpleNamespace(configured=self.configured, queued_records=len(self.pending), lost_records=0)

        def event(self, name: str, body: dict, *, attributes=None) -> None:
            super().event(name, body, attributes=attributes)
            self.pending.append(name)
            self.drain_started = name == "terminal"

    backend = DrainingBackend()
    now = [10.0]
    monkeypatch.setattr(telemetry, "_rigging", backend)
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(telemetry.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    with telemetry.process_telemetry(configured(), "driver"):
        pass

    assert backend.delivered == ["lifecycle", "terminal"]
    assert backend.shutdown_timeouts == [pytest.approx(0.02)]


def test_iris_resources_split_task_attempt_and_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_TASK_ID", "/user/job/4:7")
    monkeypatch.setenv("IRIS_WORKER_ID", "worker-1")
    monkeypatch.setenv("IRIS_MULTIGPU_PROCESS_INDEX", "2")

    assert {
        "job_id": "/user/job",
        "task_id": "/user/job/4",
        "attempt": "7",
        "worker": "worker-1",
        "process_index": "2",
    }.items() <= telemetry._iris_resources().items()


def test_actor_restart_preserves_attempt_and_changes_actor_uid(monkeypatch: pytest.MonkeyPatch) -> None:
    import ray

    class Context:
        def __init__(self, actor_uid: str, attempt: int) -> None:
            self.actor_uid = actor_uid
            self.attempt = attempt

        def get_job_id(self):
            return "ray-job"

        def get_task_id(self):
            return "ray-task"

        def get_task_attempt_number(self):
            return self.attempt

        def get_actor_id(self):
            return self.actor_uid

        def get_node_id(self):
            return "node-1"

    contexts = iter([Context("actor-before", 0), Context("actor-after", 1)])
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: next(contexts))

    before = telemetry._ray_resources()
    after = telemetry._ray_resources()

    assert before["ray_task_attempt"] == "0"
    assert after["ray_task_attempt"] == "1"
    assert before["actor_uid"] == "actor-before"
    assert after["actor_uid"] == "actor-after"


@pytest.mark.asyncio
async def test_rollout_buffer_applies_backpressure_and_reports_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    put_started = asyncio.Event()

    class ObservedQueue(asyncio.Queue):
        async def put(self, item) -> None:
            put_started.set()
            await super().put(item)

    records: list[tuple[int, int]] = []
    monkeypatch.setattr(fully_async_trainer, "record_rollout_buffer", lambda depth, capacity: records.append((depth, capacity)))
    buffer = ObservedQueue(maxsize=1)
    first = GeneratedOutputGroup({}, "first", 1)
    second = GeneratedOutputGroup({}, "second", 1)
    await buffer.put(first)

    blocked_put = asyncio.create_task(_put_rollout_with_backpressure(buffer, second))
    await put_started.wait()
    assert not blocked_put.done()
    assert await buffer.get() is first
    await blocked_put

    assert await buffer.get() is second
    assert records == [(1, 1)]
