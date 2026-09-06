"""Rollout accounting through emitted telemetry and controlled async boundaries."""

import asyncio
import json
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Event
from types import SimpleNamespace

import pytest
import zstandard
from rigging.telemetry import serialization

from skyrl_train import rollout_observability as rollout
from skyrl_train import telemetry as training_telemetry
from skyrl_train.timing_observability import publish_step_timings


@pytest.mark.parametrize("reasons", [["length", "stop", None], None])
def test_consumed_stop_fraction_is_absent_without_complete_coverage(reasons):
    metrics = rollout.consumed_stop_metrics(reasons, 3)
    assert "consumed/length_stop_fraction" not in metrics
    assert metrics["consumed/known_stop_count"] + metrics["consumed/unknown_stop_count"] == 3
    assert metrics["consumed/stop_reason_coverage"] == (0 if reasons is None else pytest.approx(2 / 3))


def test_consumed_stop_metrics_reject_misaligned_reasons():
    with pytest.raises(ValueError, match="align"):
        rollout.consumed_stop_metrics(["length"], 2)
    assert "consumed/length_stop_fraction" not in rollout.consumed_stop_metrics([], 0)


@dataclass
class ManualClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, duration: float) -> None:
        self.value += duration


@dataclass
class Records:
    metrics: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def event(self, name, body, *, attributes):
        fields = serialization.event_fields(body, budget=16_384)
        serialization.validate_attributes(attributes)
        self.events.append({"name": name, "body": fields, "attributes": dict(attributes)})

    def select(self, name, **attributes):
        return [
            row["value"]
            for row in self.metrics
            if row["name"] == name and all(row["attributes"].get(key) == value for key, value in attributes.items())
        ]


@dataclass
class Instrument:
    name: str
    records: Records

    def record(self, value, *, attributes):
        serialization.validate_attributes(attributes)
        self.records.metrics.append({"name": self.name, "value": value, "attributes": dict(attributes)})

    def add(self, value, *, attributes):
        self.record(value, attributes=attributes)


@pytest.fixture
def records(monkeypatch):
    sink = Records()
    for name in (
        "phase_duration",
        "wait_seconds",
        "wait_count",
        "call_count",
        "buffer_dwell",
        "group_count",
        "group_tokens",
        "event_loop_lag",
    ):
        monkeypatch.setattr(rollout, name, Instrument(name, sink))
    for name in ("training_metric", "nonfinite_training_metric", "work_completed"):
        monkeypatch.setattr(training_telemetry, name, Instrument(name, sink))
    monkeypatch.setattr(rollout.telemetry, "event", sink.event)

    def reject_flush(*args, **kwargs):
        pytest.fail("A rollout publication must enqueue records without a synchronous exporter flush")

    monkeypatch.setattr(rollout.telemetry, "flush", reject_flush)
    return sink


@pytest.mark.parametrize("failure", [False, True])
def test_phase_window_preserves_independent_wall_clock_and_failure(records, monkeypatch, failure):
    monotonic = ManualClock(10)
    unix_ns = ManualClock(1_000_000_000)
    monkeypatch.setattr(rollout.time, "perf_counter", monotonic)
    monkeypatch.setattr(rollout.time, "time_ns", unix_ns)

    def execute():
        with rollout.async_phase_window("training", step=7, enabled=True):
            monotonic.advance(2)
            # A clock adjustment remains visible instead of being confused with
            # an actual negative training duration in snapshot alignment.
            unix_ns.advance(-100_000_000)
            if failure:
                raise RuntimeError("training failed")

    if failure:
        with pytest.raises(RuntimeError, match="training failed"):
            execute()
    else:
        execute()
    event = records.events[-1]
    assert event["name"] == "async_phase_window"
    assert event["body"] == {"started_unix_ms": 1000, "finished_unix_ms": 900, "duration_seconds": 2}
    assert event["attributes"]["outcome"] == ("failure" if failure else "success")
    assert event["attributes"]["step"] == "7"


@pytest.mark.asyncio
async def test_overlapping_rollout_calls_keep_independent_walls_and_identity(records):
    clock = ManualClock()
    ready = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    async def produce(index):
        with rollout.observe_rollout_call(step=index, mode="async", enabled=True, clock=clock):
            with rollout.rollout_phase("collect"), rollout.rollout_wait("engine_await"):
                ready[index].set()
                await release[index].wait()
            with rollout.rollout_phase("assemble" if index == 0 else "finalize"):
                clock.advance(index + 1)

    tasks = [asyncio.create_task(produce(index)) for index in range(2)]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), timeout=1)
        clock.advance(4)
        release[0].set()
        await asyncio.wait_for(tasks[0], timeout=1)
        clock.advance(2)
        release[1].set()
        await asyncio.wait_for(tasks[1], timeout=1)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    calls = {event["attributes"]["step"]: event for event in records.events}
    assert {key: value for key, value in calls["0"]["body"].items() if key.startswith("duration_")} == {
        "duration_rollout_call": 5.0,
        "duration_collect": 4.0,
        "duration_assemble": 1.0,
        "duration_rollout_call_residual": 0.0,
    }
    assert {key: value for key, value in calls["1"]["body"].items() if key.startswith("duration_")} == {
        "duration_rollout_call": 9.0,
        "duration_collect": 7.0,
        "duration_finalize": 2.0,
        "duration_rollout_call_residual": 0.0,
    }
    assert calls["0"]["body"]["call_id"] != calls["1"]["body"]["call_id"]
    assert records.select("wait_seconds", wait="engine_await", stat="sum", step="0") == [4.0]
    assert records.select("wait_seconds", wait="engine_await", stat="sum", step="1") == [7.0]
    assert all("call_id" not in row["attributes"] for row in records.metrics)
    previous = deepcopy(records.metrics)
    with rollout.rollout_phase("collect"):
        clock.advance(1)
    assert records.metrics == previous


@pytest.mark.asyncio
async def test_concurrent_wait_totals_exceed_call_wall_without_becoming_exclusive_spans(records):
    clock = ManualClock()
    ready = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    async def wait(index):
        with rollout.rollout_wait("engine_await"):
            ready[index].set()
            await release[index].wait()

    with rollout.observe_rollout_call(step=4, mode="async", enabled=True, clock=clock):
        with rollout.rollout_phase("collect"):
            tasks = [asyncio.create_task(wait(index)) for index in range(2)]
            try:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), timeout=1)
                clock.advance(5)
                release[0].set()
                await asyncio.wait_for(tasks[0], timeout=1)
                clock.advance(3)
                release[1].set()
                await asyncio.wait_for(tasks[1], timeout=1)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    assert records.select("wait_seconds", wait="engine_await", stat="sum") == [13.0]
    assert records.select("wait_seconds", wait="engine_await", stat="max") == [8.0]
    assert records.select("wait_count", wait="engine_await") == [2]
    assert records.select("phase_duration", phase="rollout_call") == [8.0]
    assert records.select("phase_duration", phase="rollout_call_residual") == [0.0]


def test_nested_exclusive_phase_exposes_signed_residual_and_nested_tokenization_parent(records):
    clock = ManualClock()
    with rollout.observe_rollout_call(step=1, mode="async", enabled=True, clock=clock):
        with rollout.rollout_phase("collect"):
            rollout.time_tokenization(clock.advance, 2)
            with rollout.rollout_phase("assemble"):
                clock.advance(3)
            clock.advance(2)

    assert records.select("phase_duration", phase="rollout_call_residual") == [-3.0]
    assert records.select("phase_duration", phase="rollout_tokenize", parent="rollout_collect") == [2.0]
    assert records.select("phase_duration", phase="rollout_collect", parent="rollout_call") == [7.0]


@pytest.mark.parametrize("enabled", [False, True])
def test_rollout_failure_propagates_with_optional_terminal_record(records, enabled):
    clock = ManualClock()
    with pytest.raises(ValueError, match="invalid rollout"):
        with rollout.observe_rollout_call(step=1, mode="async", enabled=enabled, clock=clock):
            with rollout.rollout_phase("collect"):
                clock.advance(2)
                raise ValueError("invalid rollout")

    assert records.select("call_count", outcome="failure") == ([1] if enabled else [])
    assert len(records.events) == int(enabled)


@pytest.mark.asyncio
async def test_cancelled_rollout_publishes_one_terminal_outcome_and_unwinds_wait(records):
    clock = ManualClock()
    started = asyncio.Event()

    async def produce():
        with rollout.observe_rollout_call(step=2, mode="async", enabled=True, clock=clock):
            with rollout.rollout_phase("collect"), rollout.rollout_wait("engine_await"):
                started.set()
                await asyncio.Future()

    task = asyncio.create_task(produce())
    await asyncio.wait_for(started.wait(), timeout=1)
    clock.advance(4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert records.select("call_count", outcome="cancelled") == [1]
    assert len(records.events) == 1
    assert records.events[0]["attributes"]["outcome"] == "cancelled"
    assert records.select("wait_seconds", wait="engine_await", stat="sum") == [4.0]


class ControlledExecutor(Executor):
    """Hold an executor submission until the test releases it through the real asyncio adapter."""

    def __init__(self):
        self.submitted = asyncio.Event()
        self.future = Future()
        self.call = None

    def submit(self, fn, /, *args, **kwargs):
        self.call = lambda: fn(*args, **kwargs)
        self.submitted.set()
        return self.future

    def complete(self):
        assert self.future.set_running_or_notify_cancel()
        self.future.set_result(self.call())


@pytest.mark.asyncio
async def test_environment_wait_separates_executor_queue_execution_and_driver_resume(records):
    clock = ManualClock()
    executor = ControlledExecutor()

    def env_step():
        clock.advance(5)
        return {"reward": 1.0}

    async def produce():
        with rollout.observe_rollout_call(step=2, mode="async", enabled=True, clock=clock):
            return await rollout.run_environment(executor, env_step)

    task = asyncio.create_task(produce())
    await asyncio.wait_for(executor.submitted.wait(), timeout=1)
    clock.advance(3)
    executor.complete()
    clock.advance(7)
    assert await asyncio.wait_for(task, timeout=1) == {"reward": 1.0}
    for name, duration in (("env_queue", 3), ("env_exec", 5), ("env_resume", 7), ("env_await", 15)):
        assert records.select("wait_seconds", wait=name, stat="sum") == [duration]
        assert records.select("wait_count", wait=name) == [1]


@pytest.mark.asyncio
async def test_cancelled_environment_thread_does_not_mutate_published_call(records):
    clock = ManualClock()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finish = Event()
    executor = ThreadPoolExecutor(max_workers=1)

    def env_step():
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(timeout=5)
        return 1.0

    async def produce():
        with rollout.observe_rollout_call(step=3, mode="async", enabled=True, clock=clock):
            return await rollout.run_environment(executor, env_step)

    task = asyncio.create_task(produce())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        clock.advance(3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        published = deepcopy(records)
        clock.advance(5)
    finally:
        finish.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(executor.shutdown, wait=True)

    assert records == published
    assert records.select("call_count", outcome="cancelled") == [1]
    assert records.select("wait_seconds", wait="env_await", stat="sum") == [3.0]
    assert records.select("wait_seconds", wait="env_exec", stat="sum") == []


def test_training_metrics_preserve_selected_values_and_count_nonfinite_values(records):
    training_telemetry.record_training_metrics(
        {
            "policy/entropy": 1.25,
            "reward/mean": -0.5,
            "async/staleness_max": 2,
            "generate/avg_num_tokens": 512.0,
            "generate/tis/exact_match_fraction": 1.0,
            "val/accuracy": 0.75,
            "eval/all/avg_score": 0.625,
            "eval/all/pass_at_N": 0.875,
            "policy/loss": float("nan"),
            "policy/grad_norm": float("inf"),
            "policy/details": [1, 2],
            "unselected/value": 99.0,
        },
        step=8,
        kind="train",
    )

    assert {
        row["attributes"]["metric"]: row["value"] for row in records.metrics if row["name"] == "training_metric"
    } == {
        "policy/entropy": 1.25,
        "reward/mean": -0.5,
        "async/staleness_max": 2.0,
        "generate/avg_num_tokens": 512.0,
        "generate/tis/exact_match_fraction": 1.0,
        "val/accuracy": 0.75,
        "eval/all/avg_score": 0.625,
        "eval/all/pass_at_N": 0.875,
    }
    assert {
        row["attributes"]["metric"]: row["value"]
        for row in records.metrics
        if row["name"] == "nonfinite_training_metric"
    } == {"policy/loss": 1, "policy/grad_norm": 1}
    assert all(row["attributes"]["step"] == "8" and row["attributes"]["phase"] == "train" for row in records.metrics)


def test_consumed_work_records_distinct_native_deltas(records):
    training_telemetry.record_consumed_work(sequences=4, response_tokens=100, loss_tokens=80, step=3)
    training_telemetry.record_consumed_work(sequences=4, response_tokens=70, loss_tokens=60, step=4)

    assert records.select("work_completed", work_kind="consumed_sample") == [4, 4]
    assert records.select("work_completed", work_kind="consumed_response_token") == [100, 70]
    assert records.select("work_completed", work_kind="consumed_loss_token") == [80, 60]
    assert {row["attributes"]["step"] for row in records.metrics} == {"3", "4"}


def test_model_interval_details_stay_bounded_without_truncating_wait_totals(records):
    clock = ManualClock(123456789.1234567)
    with rollout.observe_rollout_call(step=1, mode="async", enabled=True, clock=clock):
        with rollout.rollout_phase("collect"):
            for _ in range(200):
                with rollout.rollout_wait("model_client_await"):
                    clock.advance(0.125)

    fields = records.events[0]["body"]
    intervals = json.loads(fields["model_awaits_json"])
    assert len(fields["model_awaits_json"].encode()) <= 4096
    assert fields["interval_count"] == 200
    assert fields["truncated"] is True
    assert 0 < len(intervals) < 200
    assert records.select("wait_seconds", wait="model_client_await", stat="sum") == [25.0]
    assert records.select("wait_count", wait="model_client_await") == [200]


@pytest.mark.asyncio
async def test_event_loop_lag_monitor_records_delayed_tick_and_cancels_without_catchup(records):
    clock = ManualClock()
    tick_started = asyncio.Event()
    release_tick = asyncio.Event()
    step = 2

    async def wait(interval):
        tick_started.set()
        await release_tick.wait()
        release_tick.clear()

    task = asyncio.create_task(rollout.monitor_event_loop_lag(step_fn=lambda: step, clock=clock, wait=wait))
    try:
        await asyncio.wait_for(tick_started.wait(), timeout=1)
        tick_started.clear()
        clock.advance(3.5)
        step = 3
        release_tick.set()
        await asyncio.wait_for(tick_started.wait(), timeout=1)
        assert records.select("event_loop_lag", step="3") == [2.5]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert records.select("event_loop_lag") == [2.5]


@pytest.mark.asyncio
async def test_rollout_calls_progress_while_real_exporter_waits_for_http_ack(monkeypatch):
    exporter = rollout.telemetry
    exporter.shutdown(timeout=0)
    loop = asyncio.get_running_loop()
    request_started = asyncio.Event()
    acknowledge = Event()
    delivered = []

    def post(session, endpoint, *, data, headers, timeout):
        if headers.get("Content-Encoding") == "zstd":
            data = zstandard.ZstdDecompressor().decompress(data)
        envelope = json.loads(data)
        loop.call_soon_threadsafe(request_started.set)
        assert acknowledge.wait(timeout=5)
        delivered.extend(envelope["records"])
        return SimpleNamespace(
            status_code=200,
            headers={},
            json=lambda: {"batch_id": envelope["batch_id"], "status": "accepted"},
        )

    monkeypatch.setattr(exporter.requests.Session, "post", post)
    flush = exporter.flush

    def reject_flush(*args, **kwargs):
        pytest.fail("Rollout publication synchronously flushed the exporter")

    monkeypatch.setattr(exporter, "flush", reject_flush)
    owner = training_telemetry.ProcessTelemetry(
        training_telemetry.TelemetryConfig(
            endpoint="http://finelog.test/v1/ingest", run_id="rollout-test", execution_uid="test-attempt"
        ),
        role="trainer",
    )
    owner.__enter__()
    clock = ManualClock()
    try:
        with rollout.observe_rollout_call(step=1, mode="async", enabled=True, clock=clock):
            with rollout.rollout_phase("collect"):
                clock.advance(2)
        await asyncio.wait_for(request_started.wait(), timeout=1)

        async def produce_next():
            with rollout.observe_rollout_call(step=2, mode="async", enabled=True, clock=clock):
                with rollout.rollout_phase("collect"):
                    clock.advance(3)
            return "next rollout completed"

        assert await asyncio.wait_for(produce_next(), timeout=1) == "next rollout completed"
        publish_step_timings({"step": 5.0, "policy_train": 2.0}, step=2)
        rollout.record_group_outcome(outcome="consumed", tokens=20, step=2, attempt_id="group-1")
        training_telemetry.record_training_metrics(
            {
                **rollout.consumed_stop_metrics(["length", "stop", "length", "stop"], 4),
                "tis/batch_skipped_no_logprobs": 1.0,
                "tis/skipped_fraction": 0.25,
                "unselected/value": 99.0,
            },
            step=2,
            kind="train",
        )
        assert exporter.runtime_status().queued_records > 0
        acknowledge.set()
        assert await asyncio.to_thread(flush, timeout=2)
        assert exporter.runtime_status().lost_records == 0
    finally:
        acknowledge.set()
        await asyncio.to_thread(owner.__exit__, None, None, None)

    calls = [row for row in delivered if row["name"] == "rollout_call" and row["kind"] == "event"]
    assert {row["attributes"]["step"] for row in calls} == {"1", "2"}
    assert len(calls) == 2
    root_spans = [
        row
        for row in delivered
        if row["name"] == "phase_duration_seconds" and row["attributes"]["phase"] in {"step", "rollout_call"}
    ]
    assert len(root_spans) == 3
    assert all("parent" not in row["attributes"] for row in root_spans)
    assert any(row["name"] == "rollout_group_outcome" and row["body"]["call_id"] == "group-1" for row in delivered)
    metrics = [row for row in delivered if row["name"] == "training_metric_value"]
    assert {row["attributes"]["metric"]: row["value"] for row in metrics} == {
        "consumed/sequences": 4,
        "consumed/length_stop_count": 2,
        "consumed/known_stop_count": 4,
        "consumed/unknown_stop_count": 0,
        "consumed/stop_reason_coverage": 1,
        "consumed/length_stop_fraction": 0.5,
        "tis/batch_skipped_no_logprobs": 1.0,
        "tis/skipped_fraction": 0.25,
    }
    assert all(
        row["attributes"]["step"] == "2"
        and row["attributes"]["phase"] == "train"
        and row["attributes"]["role"] == "trainer"
        for row in metrics
    )
    terminal = next(row for row in delivered if row["name"] == "terminal")
    assert terminal["body"]["status"] == "completed"
    assert terminal["body"]["export_lost_records"] == 0
