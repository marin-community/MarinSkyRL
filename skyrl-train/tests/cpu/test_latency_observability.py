from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import asyncio
import copy
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest

import skyrl_train.telemetry as trainer_telemetry
from skyrl_train.retention_observability import GenerationRetentionObserver, estimate_group_payload
from skyrl_train.trajectory_runners.harbor import observability as harbor_observability


class _Recorder:
    def __init__(self):
        self.calls = []

    def add(self, value, *, attributes):
        self.calls.append((value, attributes))

    set = add
    record = add


class _WeakGroup:
    pass


def _representative_trajectory_batch():
    return {
        "prompt_token_ids": [[1, 2, 3], [4, 5]],
        "response_ids": [[6, 7], [8, 9, 10]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1, 1], [1, 0, 1]],
        "rollout_logprobs": [[-0.1, -0.2], [-0.3, -0.4, -0.5]],
        "rollout_routed_experts": [
            [[list(range(8)) for _ in range(4)] for _ in range(2)],
            [[list(range(8)) for _ in range(4)] for _ in range(3)],
        ],
        "rollout_metrics": {"latency": 1.25},
        "is_last_step": [True, True],
        "actual_global_step": 14,
    }


def test_retention_observer_preserves_representation_and_does_not_retain_groups():
    batch = _representative_trajectory_batch()
    original = copy.deepcopy(batch)
    field_identities = {name: id(value) for name, value in batch.items()}
    group = _WeakGroup()
    group_ref = weakref.ref(group)
    observer = GenerationRetentionObserver(publish_interval_seconds=3600)

    estimate = observer.register_group(
        group,
        trajectory_batch=batch,
        source_prompts=[{"uid": "dataset-row", "prompt": "hello"}],
        owner="producer",
    )
    observer.transfer_group(group, owner="completed_buffer")
    snapshot = observer.snapshot()

    assert batch == original
    assert {name: id(value) for name, value in batch.items()} == field_identities
    assert estimate.rows == 2
    assert estimate.total_bytes > 0
    assert snapshot.groups == {"completed_buffer": 1}
    assert snapshot.rows == {"completed_buffer": 2}
    assert snapshot.estimated_bytes["completed_buffer"] == estimate.total_bytes

    del group
    gc.collect()
    assert group_ref() is None


@pytest.mark.asyncio
async def test_retention_observer_keeps_heartbeat_for_overlapping_leases():
    snapshots = []
    observer = GenerationRetentionObserver(
        publish_interval_seconds=3600,
        publish=lambda snapshot, boundary: snapshots.append((snapshot, boundary)),
    )

    observer.start()
    observer.start()

    await observer.stop()

    assert [boundary for _, boundary in snapshots] == ["observer_started"]

    await observer.stop()

    assert [boundary for _, boundary in snapshots] == ["observer_started", "observer_stopped"]


@pytest.mark.asyncio
async def test_retention_observer_restart_does_not_clear_new_heartbeat_generation():
    snapshots = []

    def publish(snapshot, boundary):
        snapshots.append((snapshot, boundary))
        if boundary == "observer_stopped" and sum(item == "observer_stopped" for _, item in snapshots) <= 2:
            observer.start()

    observer = GenerationRetentionObserver(
        publish_interval_seconds=3600,
        publish=publish,
    )

    observer.start()
    observer.start()
    await asyncio.gather(observer.stop(), observer.stop())
    await observer.stop()
    await observer.stop()

    assert [boundary for _, boundary in snapshots] == [
        "observer_started",
        "observer_stopped",
        "observer_started",
        "observer_stopped",
        "observer_started",
        "observer_stopped",
    ]


def test_payload_estimator_is_bounded_and_preserves_deep_input():
    batch = _representative_trajectory_batch()
    batch["rollout_routed_experts"] = [[[list(range(16)) for _ in range(48)] for _ in range(256)] for _ in range(2)]
    original = copy.deepcopy(batch)

    estimate = estimate_group_payload(batch, [{"uid": "dataset-row"}], max_nodes_per_field=64)

    assert batch == original
    assert estimate.rows == 2
    assert estimate.total_bytes > 0
    assert estimate.sampled_nodes <= 64 * (len(batch) + 1)
    assert estimate.truncated


def test_retention_metrics_publish_low_cardinality_snapshot(monkeypatch):
    recorders = {
        name: _Recorder()
        for name in (
            "generation_retained_groups",
            "generation_retained_rows",
            "generation_retained_estimated_bytes",
            "generation_producers",
            "generation_retention_events",
            "process_memory_bytes",
            "generation_memory_boundaries",
        )
    }
    for name, recorder in recorders.items():
        monkeypatch.setattr(trainer_telemetry, name, recorder)
    monkeypatch.setattr(
        trainer_telemetry,
        "_process_memory_snapshot",
        lambda: {"rss": 100, "uss": 80, "pss": 90, "system_available": 1_000},
    )

    trainer_telemetry.record_generation_retention_snapshot(
        groups={"producer": 2, "completed_buffer": 3},
        rows={"producer": 16, "completed_buffer": 24},
        estimated_bytes={"producer": 200, "completed_buffer": 300},
        field_estimated_bytes={
            ("producer", "response_ids"): 120,
            ("completed_buffer", "rollout_routed_experts"): 250,
        },
        producers={"generating": 5, "blocked_on_buffer": 1},
        settled_groups=7,
        settled_rows=56,
        boundary="admission_complete",
    )

    assert (3, {"owner": "completed_buffer"}) in recorders["generation_retained_groups"].calls
    assert (250, {"owner": "completed_buffer", "field": "rollout_routed_experts"}) in recorders[
        "generation_retained_estimated_bytes"
    ].calls
    assert (1, {"state": "blocked_on_buffer"}) in recorders["generation_producers"].calls
    assert (100, {"kind": "rss"}) in recorders["process_memory_bytes"].calls
    assert recorders["generation_memory_boundaries"].calls == [(1, {"boundary": "admission_complete"})]


def test_tracker_forwards_only_finite_numeric_metrics_with_step(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(trainer_telemetry, "tracker_metric", recorder)
    monkeypatch.setattr(trainer_telemetry, "record_telemetry_health", lambda: None)

    emitted = trainer_telemetry.record_tracker_metrics(
        {"loss": 1.25, "count": 3, "flag": True, "name": "x", "nan": float("nan"), "inf": float("inf")},
        step=7,
    )

    assert emitted == 2
    assert recorder.calls == [
        (1.25, {"metric": "loss", "step": "7", "role": "trainer"}),
        (3.0, {"metric": "count", "step": "7", "role": "trainer"}),
    ]


def test_generation_input_coverage_deduplicates_expanded_samples(monkeypatch):
    recorder = _Recorder()
    health = []
    monkeypatch.setattr(trainer_telemetry, "generation_input_coverage", recorder)
    monkeypatch.setattr(trainer_telemetry, "record_telemetry_health", lambda: health.append(True))
    gym = {"extra_info": {"nemotron_ultra": {"blend": "rlvr1", "agent": "math", "route": "gym"}}}
    harbor = {"extra_info": {"nemotron_ultra": {"blend": "rlvr2", "agent": "swe", "route": "terminal_bench"}}}

    emitted = trainer_telemetry.record_generation_input_coverage([gym, gym, harbor, harbor, {}])

    assert emitted == 2
    assert sorted(recorder.calls, key=lambda call: call[1]["route"]) == [
        (1, {"blend": "rlvr1", "agent": "math", "route": "gym"}),
        (1, {"blend": "rlvr2", "agent": "swe", "route": "harbor"}),
    ]
    assert health == [True]


def test_process_lifecycle_uses_bounded_event_bodies(monkeypatch):
    class EventBody:
        def __init__(self, fields):
            assert all(value is not None for value in fields.values())
            self.fields = fields

    events = []
    health = []
    shutdowns = []
    fake_telemetry = SimpleNamespace(
        serialization=SimpleNamespace(EventBody=EventBody),
        configure=lambda **_kwargs: None,
        runtime_status=lambda: SimpleNamespace(
            configured=True,
            queued_records=0,
            lost_records=0,
        ),
        event=lambda name, body, *, attributes: events.append((name, body, attributes)),
        record_runtime_health=lambda: health.append(True),
        shutdown=lambda timeout: shutdowns.append(timeout),
    )
    smoke = _Recorder()
    monkeypatch.setattr(trainer_telemetry, "telemetry", fake_telemetry)
    monkeypatch.setattr(trainer_telemetry, "telemetry_smoke", smoke)

    config = trainer_telemetry.TelemetryConfig(endpoint="http://telemetry", run_id="run", execution_uid="exec")
    with trainer_telemetry.ProcessTelemetry(config, trainer_telemetry.TRAINER_ROLE):
        pass

    assert [(name, body.fields, attributes) for name, body, attributes in events] == [
        ("lifecycle", {"state": "started"}, {"role": "trainer"}),
        (
            "terminal",
            {
                "status": "completed",
                "reason": "normal_exit",
                "export_queued_records": 0,
                "export_lost_records": 0,
            },
            {"role": "trainer"},
        ),
    ]
    assert smoke.calls == [(1, {"role": "trainer", "state": "started"})]
    assert len(health) == 2
    assert shutdowns == [trainer_telemetry.SHUTDOWN_TIMEOUT_SECONDS]


@pytest.mark.asyncio
async def test_executor_cancellation_preserves_work_cleanup(monkeypatch):
    work = _Recorder()
    cancellations = _Recorder()
    monkeypatch.setattr(trainer_telemetry, "executor_work", work)
    monkeypatch.setattr(trainer_telemetry, "executor_cancellations", cancellations)
    monkeypatch.setattr(trainer_telemetry, "executor_queue_delay", _Recorder())
    monkeypatch.setattr(trainer_telemetry, "executor_duration", _Recorder())
    executor = ThreadPoolExecutor(max_workers=1)
    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def blocker():
        blocker_started.set()
        release_blocker.wait(2)

    executor.submit(blocker)
    assert blocker_started.wait(1)
    task = asyncio.create_task(trainer_telemetry.run_in_executor_observed(executor, "environment", lambda: 7))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_blocker.set()
    await asyncio.to_thread(executor.shutdown)

    assert cancellations.calls == [(1, {"work_kind": "environment"})]
    final = {attributes["state"]: value for value, attributes in work.calls}
    assert final == {"queued": 0, "active": 0}


@pytest.mark.asyncio
async def test_harbor_lifecycle_cancellation_missing_phase_and_heartbeat(monkeypatch):
    recorders = {
        name: _Recorder()
        for name in (
            "transition_count",
            "stage_work",
            "pending_age",
            "last_progress_age",
            "trial_phase_duration",
            "trial_results",
            "trial_retries",
            "trial_tokens",
        )
    }
    for name, recorder in recorders.items():
        monkeypatch.setattr(harbor_observability, name, recorder)
    monkeypatch.setattr(harbor_observability, "record_telemetry_health", lambda: None)

    observer = harbor_observability.HarborLifecycleObserver(heartbeat_seconds=3600)
    observer.start()
    observer.register_submitted(2)
    await observer(SimpleNamespace(event=SimpleNamespace(value="start"), trial_id="t1", task_name="task", result=None))
    await observer(
        SimpleNamespace(event=SimpleNamespace(value="agent_start"), trial_id="t1", task_name="task", result=None)
    )
    await observer(SimpleNamespace(event=SimpleNamespace(value="cancel"), trial_id="t1", task_name="task", result=None))

    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    result = SimpleNamespace(
        id="t2",
        task_name="task",
        started_at=start,
        finished_at=start + timedelta(seconds=9),
        environment_setup=SimpleNamespace(started_at=start, finished_at=start + timedelta(seconds=2)),
        agent_setup=None,
        agent_execution=None,
        verifier=None,
        exception_info=None,
        compute_token_cost_totals=lambda: (10, 3, 4, None),
    )
    await observer(SimpleNamespace(event=SimpleNamespace(value="start"), trial_id="t2", task_name="task", result=None))
    await observer(
        SimpleNamespace(
            event=SimpleNamespace(value="result_persistence_start"), trial_id="t2", task_name="task", result=result
        )
    )
    await observer(SimpleNamespace(event=SimpleNamespace(value="end"), trial_id="t2", task_name="task", result=result))
    failed_result = SimpleNamespace(
        **{**result.__dict__, "id": "t3", "exception_info": SimpleNamespace(exception_type="SandboxError")}
    )
    await observer(SimpleNamespace(event=SimpleNamespace(value="start"), trial_id="t3", task_name="task", result=None))
    await observer(
        SimpleNamespace(
            event=SimpleNamespace(value="result_persistence_start"),
            trial_id="t3",
            task_name="task",
            result=failed_result,
        )
    )
    await observer(
        SimpleNamespace(event=SimpleNamespace(value="end"), trial_id="t3", task_name="task", result=failed_result)
    )
    await observer(SimpleNamespace(event=SimpleNamespace(value="start"), trial_id="t3", task_name="task", result=None))
    await observer(SimpleNamespace(event=SimpleNamespace(value="cancel"), trial_id="t3", task_name="task", result=None))
    snapshot = observer.snapshot()
    await observer.stop()

    assert snapshot["pending"] == 0
    assert {attributes["stage"] for _, attributes in recorders["transition_count"].calls} >= {
        "start",
        "agent_start",
        "cancel",
        "result_persistence_start",
        "end",
    }
    phases = {attributes["phase"] for _, attributes in recorders["trial_phase_duration"].calls}
    assert phases == {"environment_setup", "total"}
    assert recorders["trial_results"].calls == [
        (1, {"outcome": "success", "exception_class": "none"}),
        (1, {"outcome": "failure", "exception_class": "SandboxError"}),
    ]
    assert recorders["trial_retries"].calls == [(1, {"reason": "SandboxError", "subsystem": "harbor"})]


def test_harbor_group_tail_math_ignores_exception_results(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(harbor_observability, "group_tail", recorder)
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)

    def result(total: int, agent: int):
        return SimpleNamespace(
            started_at=start,
            finished_at=start + timedelta(seconds=total),
            environment_setup=None,
            agent_setup=None,
            agent_execution=SimpleNamespace(started_at=start, finished_at=start + timedelta(seconds=agent)),
            verifier=None,
        )

    harbor_observability.record_harbor_group([result(10, 7), RuntimeError("failed"), result(20, 15)])

    values = {attributes["statistic"]: value for value, attributes in recorder.calls}
    assert values["p50"] == 15
    assert values["max"] == 20
    assert values["spread"] == 10
    assert values["slowest_phase"] == 15
