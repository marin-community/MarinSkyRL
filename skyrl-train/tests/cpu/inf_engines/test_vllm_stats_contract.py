from pathlib import Path
from types import SimpleNamespace

from skyrl_train.inference_engines.vllm.stats import (
    VLLM_FINISH_REASONS,
    VLLMHistogramSnapshot,
    VLLMNativeStatsAccumulator,
    build_1_2_5_buckets,
)
from skyrl_train.inference_engines.vllm.stats import (
    HTTPBridgeStatsAccumulator,
    IntervalReadMode,
    VLLMCumulativeStats,
    VLLMEngineStatsSnapshot,
    VLLMIntervalStats,
    InferenceStatsSnapshot,
)
from skyrl_train.inference_observability import (
    PUBLICATION_LOSS_METRIC,
    VLLM_MAX_RECORDS_PER_ENGINE,
    FinelogInferenceMetricsSink,
)


class _Publisher:
    def __init__(self, *, sample_limit_dropped_records=0, telemetry_lost_records=0):
        self.calls = []
        self.result = SimpleNamespace(
            configured=True,
            enqueued_records=0,
            sample_limit_dropped_records=sample_limit_dropped_records,
            telemetry_lost_records=telemetry_lost_records,
        )

    def publish(self, records):
        self.calls.append(tuple(records))
        return self.result


def _capture_health(monkeypatch):
    from rigging import telemetry

    health = []

    class _Gauge:
        def set(self, value, *, attributes):
            health.append((value, attributes))

    monkeypatch.setattr(telemetry, "gauge", lambda name, *, unit: _Gauge() if name == PUBLICATION_LOSS_METRIC else None)
    return health


def _snapshot(engine_id="engine-a", *, attributes=None, histograms=None):
    accumulator = VLLMNativeStatsAccumulator((1, 10), {"model_name": "m", "engine_index": "0"})
    accumulator.current = accumulator.current.__class__(2, 3, 1, 0.4)
    accumulator.prompt_tokens = 20
    accumulator.generation_tokens = 12
    accumulator.prefix_cache_hits = 5
    accumulator.prefix_cache_queries = 8
    accumulator.preemptions = 1
    native = accumulator.snapshot()
    return VLLMEngineStatsSnapshot(
        engine_id,
        1.0,
        native.current,
        native.cumulative,
        VLLMIntervalStats(),
        attributes=attributes or {"model_name": "m", "engine_index": "0"},
        histograms=native.histograms if histograms is None else histograms,
    )


def test_http_bridge_accumulator_peek_and_reset_interval_without_resetting_histograms():
    accumulator = HTTPBridgeStatsAccumulator()
    labels = {"endpoint": "/chat/completions", "transport": "json", "status": "2xx"}
    accumulator.observe("response_bytes", 100, attributes=labels)
    accumulator.observe("response_bytes", 300, attributes=labels)

    peek = accumulator.snapshot(IntervalReadMode.PEEK)
    reset = accumulator.snapshot(IntervalReadMode.RESET)
    empty_interval = accumulator.snapshot(IntervalReadMode.PEEK)

    assert peek.response_bytes.count == reset.response_bytes.count == 2
    assert peek.response_bytes.mean == 200
    assert empty_interval.response_bytes.count == 0
    assert empty_interval.histograms[0].count == 2


def test_native_accumulator_preserves_current_cumulative_labels_and_histograms():
    accumulator = VLLMNativeStatsAccumulator(build_1_2_5_buckets(100), {"model_name": "m", "engine_index": "0"})
    scheduler = SimpleNamespace(
        num_running_reqs=2,
        num_waiting_reqs=3,
        num_skipped_waiting_reqs=1,
        kv_cache_usage=0.4,
        prefix_cache_stats=SimpleNamespace(hits=5, queries=8),
    )
    finished = SimpleNamespace(
        finish_reason="length",
        queued_time=0.1,
        prefill_time=0.2,
        decode_time=0.7,
        e2e_latency=1.0,
        num_generation_tokens=12,
        mean_time_per_output_token=0.07,
    )
    iteration = SimpleNamespace(
        num_prompt_tokens=20,
        num_generation_tokens=12,
        num_preempted_reqs=1,
        prompt_token_stats=SimpleNamespace(computed=7),
        time_to_first_tokens_iter=[0.3],
        finished_requests=[finished],
    )

    accumulator.observe(scheduler, iteration)
    native = accumulator.snapshot()

    assert (native.current.running_requests, native.current.waiting_capacity, native.current.waiting_deferred) == (
        2,
        3,
        1,
    )
    assert native.cumulative.prompt_tokens == 20
    assert native.cumulative.generation_tokens == 12
    assert native.cumulative.finished_by_reason == {
        "stop": 0,
        "length": 1,
        "abort": 0,
        "error": 0,
        "repetition": 0,
    }
    assert {histogram.name for histogram in native.histograms} == {
        "request_queue_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "e2e_request_latency_seconds",
        "time_to_first_token_seconds",
        "request_generation_tokens",
        "iteration_tokens_total",
        "request_time_per_output_token_seconds",
    }
    assert all(histogram.attributes == {"model_name": "m", "engine_index": "0"} for histogram in native.histograms)
    assert next(item for item in native.histograms if item.name == "request_generation_tokens").total == 12
    assert next(item for item in native.histograms if item.name == "iteration_tokens_total").total == 19
    assert (
        next(item for item in native.histograms if item.name == "request_time_per_output_token_seconds").total == 0.07
    )


def test_blank_cumulative_snapshot_exposes_every_finish_reason_at_zero():
    assert VLLMCumulativeStats().finished_by_reason == {reason: 0 for reason in VLLM_FINISH_REASONS}


def test_native_accumulator_is_cumulative_across_reads():
    accumulator = VLLMNativeStatsAccumulator((1, 10), {})
    iteration = SimpleNamespace(
        num_prompt_tokens=2,
        num_generation_tokens=3,
        num_preempted_reqs=0,
        prompt_token_stats=SimpleNamespace(computed=1),
        time_to_first_tokens_iter=[],
        finished_requests=[],
    )

    accumulator.observe(None, iteration)
    first = accumulator.snapshot()
    accumulator.observe(None, iteration)
    second = accumulator.snapshot()

    assert first.cumulative.prompt_tokens == 2
    assert second.cumulative.prompt_tokens == 4


def test_finelog_adapter_projects_every_typed_native_measurement(monkeypatch):
    snapshot = InferenceStatsSnapshot((_snapshot(),))
    publisher = _Publisher()
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = publisher
    health = _capture_health(monkeypatch)

    sink.publish(snapshot, step=7)
    (published,) = publisher.calls

    assert {
        "num_requests_running",
        "num_requests_waiting",
        "num_requests_waiting_by_reason",
        "kv_cache_usage_perc",
        "num_preemptions_total",
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "generation_tokens_total",
        "prompt_tokens_total",
        "request_success_total",
        "iteration_tokens_total_bucket",
        "iteration_tokens_total_count",
        "iteration_tokens_total_sum",
        "request_time_per_output_token_seconds_bucket",
        "request_time_per_output_token_seconds_count",
        "request_time_per_output_token_seconds_sum",
    }.issubset(record.name for record in published)
    expected_histogram_names = {
        f"{histogram.name}_{component}"
        for histogram in _snapshot().histograms
        for component in ("bucket", "count", "sum")
    }
    assert {record.name for record in published if record.source_kind == "histogram"} == expected_histogram_names
    reasons = {
        record.attributes.get("reason") for record in published if record.name == "num_requests_waiting_by_reason"
    }
    assert reasons == {"capacity", "deferred"}
    assert all(record.attributes["engine"] == "engine-a" for record in published)
    assert {
        record.attributes["finished_reason"] for record in published if record.name == "request_success_total"
    } == set(VLLM_FINISH_REASONS)
    assert all(record.attributes["step"] == "7" for record in published if record.source_kind == "gauge")
    assert all("step" not in record.attributes for record in published if record.source_kind != "gauge")
    assert health == [
        (0, {"metric_source": "vllm", "drop_reason": "sample_limit"}),
        (0, {"metric_source": "vllm", "drop_reason": "telemetry_loss"}),
    ]


def test_physical_engine_identity_survives_same_native_index_and_step_changes(monkeypatch):
    publisher = _Publisher()
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = publisher
    _capture_health(monkeypatch)
    snapshot = InferenceStatsSnapshot((_snapshot("engine-a"), _snapshot("engine-b")))

    sink.publish(snapshot, step=7)
    sink.publish(snapshot, step=8)

    histogram_batches = [
        tuple(record for record in batch if record.name == "iteration_tokens_total_count") for batch in publisher.calls
    ]
    assert [[record.attributes["engine"] for record in batch] for batch in histogram_batches] == [
        ["engine-a"],
        ["engine-b"],
        ["engine-a"],
        ["engine-b"],
    ]
    assert all(record.attributes["engine_index"] == "0" for batch in histogram_batches for record in batch)
    assert all("step" not in record.attributes for batch in histogram_batches for record in batch)


def test_oversized_engine_batch_is_rejected_whole_before_publish(monkeypatch):
    oversized = VLLMHistogramSnapshot(
        "oversized",
        tuple((float(index), 0.0) for index in range(VLLM_MAX_RECORDS_PER_ENGINE + 1)),
        0,
        0,
        "1",
    )
    publisher = _Publisher()
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = publisher
    health = _capture_health(monkeypatch)

    sink.publish(InferenceStatsSnapshot((_snapshot(histograms=(oversized,)),)), step=1)

    assert publisher.calls == []
    assert health[0][0] > VLLM_MAX_RECORDS_PER_ENGINE
    assert health[1][0] == 0


def test_invalid_engine_batch_is_rejected_whole_before_publish(monkeypatch):
    invalid = VLLMHistogramSnapshot("invalid", ((1.0, float("nan")),), 0, 0, "1")
    publisher = _Publisher()
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = publisher
    health = _capture_health(monkeypatch)

    sink.publish(InferenceStatsSnapshot((_snapshot(histograms=(invalid,)),)), step=1)

    assert publisher.calls == []
    assert health[0][0] == 0
    assert health[1][0] > 0


def test_publication_result_loss_is_exposed_as_current_health(monkeypatch):
    publisher = _Publisher(sample_limit_dropped_records=2, telemetry_lost_records=3)
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = publisher
    health = _capture_health(monkeypatch)

    sink.publish(InferenceStatsSnapshot((_snapshot(),)), step=1)

    assert health == [
        (2, {"metric_source": "vllm", "drop_reason": "sample_limit"}),
        (3, {"metric_source": "vllm", "drop_reason": "telemetry_loss"}),
    ]


def test_process_wide_http_histograms_have_no_engine_or_step_identity(monkeypatch):
    bridge = HTTPBridgeStatsAccumulator()
    bridge.observe("response_bytes", 128, attributes={"endpoint": "/chat/completions"})
    publisher = _Publisher()
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._publisher = _Publisher()
    sink._bridge_publisher = publisher
    _capture_health(monkeypatch)

    sink.publish(InferenceStatsSnapshot((), bridge.snapshot(IntervalReadMode.PEEK)), step=9)

    (records,) = publisher.calls
    assert records
    assert all("engine" not in record.attributes and "engine_index" not in record.attributes for record in records)
    assert all("step" not in record.attributes for record in records)


def test_vllm_engine_package_has_no_direct_publisher_or_parallel_prometheus_path():
    package = Path(__file__).parents[3] / "skyrl_train" / "inference_engines" / "vllm"
    forbidden = {
        "enable_ray_prometheus_stats",
        "RayPrometheusStatLogger",
        "PrometheusStatLogger",
        "MetricSnapshotPublisher",
        "prometheus_client",
        "rigging.telemetry",
        "wandb.log",
        "REGISTRY.collect",
    }

    findings = {
        (path.relative_to(package), token)
        for path in package.rglob("*.py")
        for token in forbidden
        if token in path.read_text()
    }
    assert findings == set()


def test_http_bridge_uses_the_callback_boundary_instead_of_publishing_directly():
    endpoint = (
        Path(__file__).parents[3] / "skyrl_train" / "inference_engines" / "inference_engine_client_http_endpoint.py"
    )
    source = endpoint.read_text()

    assert not {
        token
        for token in ("MetricSnapshotPublisher", "rigging.telemetry", "wandb.log", "all_metrics")
        if token in source
    }
