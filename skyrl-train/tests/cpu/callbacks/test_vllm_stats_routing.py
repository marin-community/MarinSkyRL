import asyncio
from types import SimpleNamespace

from skyrl_train.callbacks.base import TrainerControl, TrainerState
from skyrl_train.callbacks.builtin import InferenceStatsCallback
from skyrl_train.inference_engines.vllm.stats import (
    HTTPBridgeStatsAccumulator,
    IntervalReadMode,
    VLLMEngineStatsSnapshot,
    VLLMHistogramSnapshot,
    VLLMIntervalStats,
    VLLMNativeStatsAccumulator,
    InferenceStatsSnapshot,
)
from skyrl_train.inference_observability import (
    PUBLICATION_LOSS_METRIC,
    VLLM_MAX_RECORDS_PER_ENGINE,
    FinelogInferenceMetricsSink,
)


class _Publisher:
    def __init__(self, *, sample_limit=0, telemetry_loss=0):
        self.calls = []
        self.result = SimpleNamespace(
            configured=True,
            enqueued_records=0,
            sample_limit_dropped_records=sample_limit,
            telemetry_lost_records=telemetry_loss,
        )

    def publish(self, records):
        self.calls.append(tuple(records))
        return self.result


class _EngineClient:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.read_modes = []

    async def get_stats(self, read_mode):
        self.read_modes.append(read_mode)
        return next(self.snapshots)


class _Trainer:
    def __init__(self, snapshots):
        self.all_metrics = {}
        self.inference_engine_client = _EngineClient(snapshots)


def _engine_snapshot(engine_id, native, *, interval=VLLMIntervalStats(), histograms=None):
    return VLLMEngineStatsSnapshot(
        engine_id,
        1.0,
        native.current,
        native.cumulative,
        interval,
        attributes={"model_name": "m", "engine_index": "0"},
        histograms=native.histograms if histograms is None else histograms,
    )


def _capture_health(monkeypatch):
    from rigging import telemetry

    health = []

    class _Gauge:
        def set(self, value, *, attributes):
            health.append((attributes["metric_source"], attributes["drop_reason"], value))

    def gauge(name, *, unit):
        assert (name, unit) == (PUBLICATION_LOSS_METRIC, "{record}")
        return _Gauge()

    monkeypatch.setattr(telemetry, "gauge", gauge)
    return health


def test_embedded_stats_end_to_end(monkeypatch):
    accumulator = VLLMNativeStatsAccumulator((1, 10, 100), {"model_name": "m", "engine_index": "0"})
    zero = accumulator.snapshot()
    accumulator.observe(
        SimpleNamespace(
            num_running_reqs=2,
            num_waiting_reqs=3,
            num_skipped_waiting_reqs=1,
            kv_cache_usage=0.4,
            prefix_cache_stats=SimpleNamespace(hits=5, queries=8),
        ),
        SimpleNamespace(
            num_prompt_tokens=20,
            num_generation_tokens=12,
            num_preempted_reqs=1,
            prompt_token_stats=SimpleNamespace(computed=7),
            time_to_first_tokens_iter=[0.3],
            finished_requests=[
                SimpleNamespace(
                    finish_reason="length",
                    queued_time=0.1,
                    prefill_time=0.2,
                    decode_time=0.7,
                    e2e_latency=1.0,
                    num_generation_tokens=12,
                    mean_time_per_output_token=0.07,
                )
            ],
        ),
    )
    observed = accumulator.snapshot()
    interval = VLLMIntervalStats(
        peak_running_reqs=2,
        peak_waiting_reqs=3,
        median_running_reqs=2,
        median_waiting_reqs=3,
        finished_requests=1,
        preempted_reqs=1,
        samples=1,
        active_samples=1,
    )

    bridge = HTTPBridgeStatsAccumulator()
    labels = {"endpoint": "/chat/completions", "transport": "json", "status": "2xx"}
    bridge.observe("response_bytes", 100, attributes=labels)
    bridge.observe("response_bytes", 300, attributes=labels)
    initial_bridge = bridge.snapshot(IntervalReadMode.PEEK)
    reset_bridge = bridge.snapshot(IntervalReadMode.RESET)
    empty_bridge = bridge.snapshot(IntervalReadMode.PEEK)

    initial = InferenceStatsSnapshot(
        (_engine_snapshot("physical-a", zero), _engine_snapshot("physical-b", zero)),
        initial_bridge,
    )
    positive = InferenceStatsSnapshot(
        (
            _engine_snapshot("physical-a", observed, interval=interval),
            _engine_snapshot("physical-b", observed, interval=interval),
        ),
        reset_bridge,
    )
    engine_publisher = _Publisher()
    bridge_publisher = _Publisher()
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = engine_publisher
    sink._bridge_publisher = bridge_publisher
    health = _capture_health(monkeypatch)
    trainer = _Trainer((initial, positive))
    callback = InferenceStatsCallback(
        log_every_steps=2,
        log_to_console=False,
        poll_interval_seconds=0,
        sinks=(sink,),
    )
    control = TrainerControl()

    async def run_training_boundary():
        await callback.on_train_begin_async(
            TrainerState(global_step=0, epoch=0, total_steps=2, num_steps_per_epoch=2), control, trainer=trainer
        )
        await callback.on_step_end_async(
            TrainerState(global_step=1, epoch=0, total_steps=2, num_steps_per_epoch=2), control, trainer=trainer
        )
        await callback.on_step_end_async(
            TrainerState(global_step=2, epoch=0, total_steps=2, num_steps_per_epoch=2), control, trainer=trainer
        )

    asyncio.run(run_training_boundary())

    assert trainer.inference_engine_client.read_modes == [IntervalReadMode.PEEK, IntervalReadMode.RESET]
    assert trainer.all_metrics["vllm/num_engines"] == 2
    assert trainer.all_metrics["vllm/total_finished_requests"] == 2
    assert trainer.all_metrics["inference_bridge/response_bytes/mean"] == 200
    assert empty_bridge.response_bytes.count == 0
    assert empty_bridge.histograms[0].count == 2

    assert len(engine_publisher.calls) == 4
    assert [batch[0].attributes["engine"] for batch in engine_publisher.calls] == [
        "physical-a",
        "physical-b",
        "physical-a",
        "physical-b",
    ]
    zero_reasons = {
        record.attributes["finished_reason"]: record.value
        for record in engine_publisher.calls[0]
        if record.name == "request_success_total"
    }
    positive_records = engine_publisher.calls[2]
    positive_reasons = {
        record.attributes["finished_reason"]: record.value
        for record in positive_records
        if record.name == "request_success_total"
    }
    assert zero_reasons == {"stop": 0, "length": 0, "abort": 0, "error": 0, "repetition": 0}
    assert positive_reasons == {**zero_reasons, "length": 1}
    assert next(record.value for record in positive_records if record.name == "iteration_tokens_total_sum") == 19
    assert (
        next(record.value for record in positive_records if record.name == "request_time_per_output_token_seconds_sum")
        == 0.07
    )
    assert all(
        record.attributes["step"] in {"0", "2"}
        for batch in engine_publisher.calls
        for record in batch
        if record.source_kind == "gauge"
    )
    assert all(
        "step" not in record.attributes
        for batch in engine_publisher.calls
        for record in batch
        if record.source_kind != "gauge"
    )
    assert all(record.attributes["engine_index"] == "0" for batch in engine_publisher.calls for record in batch)

    assert len(bridge_publisher.calls) == 2
    assert all(
        "step" not in record.attributes and "engine" not in record.attributes
        for batch in bridge_publisher.calls
        for record in batch
    )
    assert health == [
        (source, reason, 0)
        for _ in range(2)
        for source in ("vllm", "inference_http_bridge")
        for reason in ("sample_limit", "telemetry_loss")
    ]
    native = VLLMNativeStatsAccumulator((1, 10), {"model_name": "m", "engine_index": "0"}).snapshot()
    invalid = VLLMHistogramSnapshot("invalid", ((1.0, float("nan")),), 0, 0, "1")
    oversized = VLLMHistogramSnapshot(
        "oversized",
        tuple((float(index), 0.0) for index in range(VLLM_MAX_RECORDS_PER_ENGINE + 1)),
        0,
        0,
        "1",
    )
    publisher = _Publisher(sample_limit=2, telemetry_loss=3)
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = publisher
    health = _capture_health(monkeypatch)

    sink.publish(
        InferenceStatsSnapshot(
            (
                _engine_snapshot("valid", native),
                _engine_snapshot("invalid", native, histograms=(invalid,)),
                _engine_snapshot("oversized", native, histograms=(oversized,)),
            )
        ),
        step=7,
    )

    assert len(publisher.calls) == 1
    assert {record.attributes["engine"] for record in publisher.calls[0]} == {"valid"}
    assert health == [
        ("vllm", "sample_limit", 532),
        ("vllm", "telemetry_loss", 21),
    ]
