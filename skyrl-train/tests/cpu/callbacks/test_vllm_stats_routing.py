from dataclasses import replace
from types import SimpleNamespace as Record

import pytest
from rigging import telemetry
from rigging.telemetry import metrics as rigging_metrics

from skyrl_train.inference_engines.vllm import stats as vllm
from skyrl_train.inference_observability import FinelogInferenceMetricsSink, VllmHistogramFormat, trainer_metrics


def test_vllm_stats_reach_finelog():
    labels = {"model_name": "m", "engine": "0"}

    def metric(name, value, **extra_labels):
        return Record(name=f"vllm:{name}", labels={**labels, **extra_labels}, value=value)

    native = vllm.snapshot_vllm_prometheus_metrics(
        [
            metric("num_requests_running", 2),
            metric("num_requests_waiting_by_reason", 3, reason="capacity"),
            metric("num_requests_waiting_by_reason", 1, reason="deferred"),
            metric("kv_cache_usage_perc", 0.4),
            metric("prompt_tokens", 20),
            metric("generation_tokens", 12),
            metric("prefix_cache_hits", 5),
            metric("prefix_cache_queries", 8),
            metric("num_preemptions", 1),
            metric("spec_decode_num_drafts", 10),
            metric("spec_decode_num_draft_tokens", 30),
            metric("spec_decode_num_accepted_tokens", 12),
            metric("request_success", 1, finished_reason="length"),
            Record(
                name="vllm:request_time_per_output_token_seconds",
                labels=labels,
                buckets={"0.01": 0, "0.1": 1, "+Inf": 1},
                count=1,
                sum=0.07,
            ),
            Record(name="vllm:generation_tokens", labels={**labels, "engine": "1"}, value=999),
        ],
        engine_index="0",
    )
    bridge = vllm.HTTPBridgeStatsAccumulator()
    bridge.observe("response_bytes", 100, attributes={"status": "2xx"})
    interval = vllm.VLLMIntervalStats(finished_requests=1)
    engine = vllm.VLLMEngineStatsSnapshot(
        "physical-a",
        1.0,
        native.current,
        native.cumulative,
        interval,
        {"model_name": "m", "engine_index": "0"},
        native.histograms,
    )
    snapshot = vllm.InferenceStatsSnapshot((engine,), bridge.snapshot(vllm.IntervalReadMode.RESET))

    batches = []
    published = Record(configured=True, sample_limit_dropped_records=0, telemetry_lost_records=0)
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._histogram_format = VllmHistogramFormat.SCALAR
    sink._publisher = Record(publish=lambda records: batches.append(records) or published)
    sink._bridge_publisher = Record(publish=lambda records: batches.append(records) or published)
    sink.publish(snapshot, step=7)

    engine, http = batches
    values = {record.name: record.value for record in engine}
    assert values["num_requests_running"] == 2
    assert values["num_requests_waiting"] == 4
    assert values["generation_tokens_total"] == 12
    assert values["prefix_cache_hits_total"] == 5
    assert values["spec_decode_num_drafts_total"] == 10
    assert values["spec_decode_num_draft_tokens_total"] == 30
    assert values["spec_decode_num_accepted_tokens_total"] == 12
    reasons = {r.attributes["finished_reason"]: r.value for r in engine if r.name == "request_success_total"}
    assert reasons == {"stop": 0, "length": 1, "abort": 0, "error": 0, "repetition": 0}
    assert values["request_time_per_output_token_seconds_sum"] == 0.07
    assert all(record.attributes["engine"] == "physical-a" for record in engine)
    assert all("engine" not in record.attributes for record in http)
    projected = trainer_metrics(snapshot)
    assert projected["vllm/total_finished_requests"] == 1
    assert projected["vllm/spec_decode_acceptance_rate"] == 0.4
    assert projected["vllm/spec_decode_mean_acceptance_length"] == 2.2


def test_native_output_length_and_tpot_reach_structured_publisher(monkeypatch) -> None:
    labels = {"engine": "0", "model_name": "m"}
    native = vllm.snapshot_vllm_prometheus_metrics(
        [
            Record(
                name="vllm:request_generation_tokens",
                labels=labels,
                buckets={"10": 2, "100": 3, "+Inf": 3},
                count=3,
                sum=120.0,
            ),
            Record(
                name="vllm:request_time_per_output_token_seconds",
                labels=labels,
                buckets={"0.02": 2, "0.1": 2, "+Inf": 2},
                count=2,
                sum=0.04,
            ),
        ],
        engine_index="0",
    )
    engine = vllm.VLLMEngineStatsSnapshot(
        engine_id="engine-incarnation-1",
        timestamp=1.0,
        current=native.current,
        cumulative=native.cumulative,
        interval=vllm.VLLMIntervalStats(),
        attributes={"model_name": "m", "engine_index": "0"},
        histograms=native.histograms,
        histogram_timestamp_ms=1_700_000_000_000,
        histogram_sequence=7,
    )
    monkeypatch.setattr(rigging_metrics, "HistogramSnapshot", Record, raising=False)
    monkeypatch.setattr(telemetry, "gauge", lambda name, *, unit: Record(set=lambda value, *, attributes: None))
    scalar_batches = []
    histogram_batches = []
    published = Record(configured=True, sample_limit_dropped_records=0, telemetry_lost_records=0)
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._histogram_format = VllmHistogramFormat.STRUCTURED
    sink._publisher = Record(publish=lambda records: scalar_batches.append(records) or published)
    sink._histogram_publisher = Record(publish=lambda records: histogram_batches.extend(records) or published)
    sink._bridge_publisher = Record(publish=lambda records: published)

    sink.publish(vllm.InferenceStatsSnapshot((engine,)), step=7)

    points = {point.name: point for point in histogram_batches}
    assert {
        name: (point.explicit_bounds, point.bucket_counts, point.count, point.sum, point.unit)
        for name, point in points.items()
    } == {
        "request_generation_tokens": ((10.0, 100.0), (2, 1, 0), 3, 120.0, "{token}"),
        "request_time_per_output_token_seconds": ((0.02, 0.1), (2, 0, 0), 2, 0.04, "s"),
    }
    assert all(
        point.attributes == {"model_name": "m", "engine_index": "0", "engine": "engine-incarnation-1"}
        and point.timestamp_ms == 1_700_000_000_000
        and point.producer_epoch == "engine-incarnation-1"
        and point.sequence == 7
        for point in points.values()
    )
    assert not any(record.name.endswith(("_bucket", "_count", "_sum")) for record in scalar_batches[0])


def test_native_histogram_snapshot_preserves_integer_counts_above_float_precision() -> None:
    first_bucket = (1 << 53) + 1
    native = vllm.snapshot_vllm_prometheus_metrics(
        [
            Record(
                name="vllm:request_queue_time_seconds",
                labels={"engine": "0"},
                buckets={"0.01": first_bucket, "0.1": first_bucket + 2, "+Inf": first_bucket + 5},
                count=first_bucket + 5,
                sum=42.5,
            )
        ],
        engine_index="0",
    )

    histogram = native.histograms[0]
    assert histogram.buckets == ((0.01, first_bucket), (0.1, first_bucket + 2), (float("inf"), first_bucket + 5))
    assert histogram.count == first_bucket + 5
    assert isinstance(histogram.count, int)


def test_native_histogram_rejects_invalid_family_without_losing_other_metrics() -> None:
    def histogram(name, count):
        return Record(
            name=f"vllm:{name}",
            labels={"engine": "0"},
            buckets={"0.1": count, "+Inf": count},
            count=count,
            sum=0.2,
        )

    for unsupported in (1 << 63, float(1 << 53)):
        native = vllm.snapshot_vllm_prometheus_metrics(
            [
                histogram("request_queue_time_seconds", unsupported),
                histogram("request_prefill_time_seconds", 2),
                Record(name="vllm:num_requests_running", labels={"engine": "0"}, value=3),
            ],
            engine_index="0",
        )
        assert native.histogram_dropped_count == 1
        assert [item.name for item in native.histograms] == ["request_prefill_time_seconds"]
        assert native.current.running_requests == 3

    incomplete = Record(
        name="vllm:request_queue_time_seconds",
        labels={"engine": "0"},
        buckets={"0.1": 2, "+Inf": 2},
        count=2,
    )
    native = vllm.snapshot_vllm_prometheus_metrics(
        [incomplete, histogram("request_prefill_time_seconds", 2)], engine_index="0"
    )
    assert native.histogram_dropped_count == 1
    assert [item.name for item in native.histograms] == ["request_prefill_time_seconds"]


def test_dual_vllm_sink_preserves_gauges_when_no_histogram_was_collected(monkeypatch) -> None:
    monkeypatch.setattr(rigging_metrics, "HistogramSnapshot", Record, raising=False)
    engine = vllm.VLLMEngineStatsSnapshot(
        engine_id="engine-a",
        timestamp=1.0,
        current=vllm.VLLMCurrentStats(running_requests=3),
        cumulative=vllm.VLLMCumulativeStats(),
        interval=vllm.VLLMIntervalStats(),
    )
    scalar_batches = []
    published = Record(configured=True, sample_limit_dropped_records=0, telemetry_lost_records=0)
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._histogram_format = VllmHistogramFormat.DUAL
    sink._publisher = Record(publish=lambda records: scalar_batches.append(records) or published)
    sink._histogram_publisher = Record(publish=lambda records: published)
    sink._bridge_publisher = Record(publish=lambda records: published)
    monkeypatch.setattr(telemetry, "gauge", lambda name, *, unit: Record(set=lambda value, *, attributes: None))

    sink.publish(vllm.InferenceStatsSnapshot((engine,)), step=7)

    assert [(record.name, record.value) for record in scalar_batches[0] if record.name == "num_requests_running"] == [
        ("num_requests_running", 3)
    ]


@pytest.mark.parametrize("histogram_format", [VllmHistogramFormat.STRUCTURED, VllmHistogramFormat.DUAL])
def test_structured_vllm_sink_emits_one_exact_family_without_scalar_bucket_rows(monkeypatch, histogram_format) -> None:
    # The frozen Rigging wheel is upgraded only after the new publisher is released.
    monkeypatch.setattr(rigging_metrics, "HistogramSnapshot", Record, raising=False)
    first_bucket = (1 << 53) + 1
    histogram = vllm.VLLMHistogramSnapshot(
        name="request_queue_time_seconds",
        buckets=((0.01, first_bucket), (0.1, first_bucket + 2), (float("inf"), first_bucket + 5)),
        count=first_bucket + 5,
        total=42.5,
        unit="s",
        attributes={"engine_index": "0"},
    )
    engine = vllm.VLLMEngineStatsSnapshot(
        engine_id="engine-incarnation-1",
        timestamp=1.0,
        current=vllm.VLLMCurrentStats(),
        cumulative=vllm.VLLMCumulativeStats(),
        interval=vllm.VLLMIntervalStats(),
        histograms=(histogram,),
        histogram_timestamp_ms=1_700_000_000_000,
        histogram_sequence=7,
        histogram_dropped_count=1,
    )
    malformed = replace(
        histogram,
        name="request_prefill_time_seconds",
        buckets=((float("inf"), first_bucket + 5), (0.1, first_bucket + 5)),
    )
    engine = replace(engine, histograms=(histogram, malformed))
    scalar_batches = []
    histogram_batches = []
    health = []
    monkeypatch.setattr(
        telemetry,
        "gauge",
        lambda name, *, unit: Record(
            set=lambda value, *, attributes: health.append((name, value, attributes["drop_reason"]))
        ),
    )
    published = Record(configured=True, enqueued_records=1, sample_limit_dropped_records=0, telemetry_lost_records=0)
    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._histogram_format = histogram_format
    sink._publisher = Record(publish=lambda records: scalar_batches.append(records) or published)
    sink._histogram_publisher = Record(publish=lambda records: histogram_batches.append(records) or published)
    sink._bridge_publisher = Record(publish=lambda records: published)

    sink.publish(vllm.InferenceStatsSnapshot((engine,)), step=7)

    assert len(histogram_batches) == 1
    assert len(histogram_batches[0]) == 1
    point = histogram_batches[0][0]
    assert point.name == "request_queue_time_seconds"
    assert point.explicit_bounds == (0.01, 0.1)
    assert point.bucket_counts == (first_bucket, 2, 3)
    assert point.count == first_bucket + 5
    assert point.sum == 42.5
    assert point.timestamp_ms == 1_700_000_000_000
    assert point.producer_epoch == "engine-incarnation-1"
    assert point.sequence == 7
    assert point.attributes == {"engine": "engine-incarnation-1", "engine_index": "0"}
    scalar_histograms = [
        record
        for batch in scalar_batches
        for record in batch
        if record.name.startswith(("request_queue_time_seconds_", "request_prefill_time_seconds_"))
    ]
    if histogram_format is VllmHistogramFormat.DUAL:
        assert len(scalar_histograms) == 5
        assert all(record.name.startswith("request_queue_time_seconds_") for record in scalar_histograms)
        assert all(
            record.attributes["histogram_publication_id"] == "engine-incarnation-1:7" for record in scalar_histograms
        )
    else:
        assert scalar_histograms == []
    assert ("metric_publication_dropped_records", 2, "telemetry_loss") in health


def test_structured_vllm_admission_cap_is_per_engine(monkeypatch) -> None:
    monkeypatch.setattr(rigging_metrics, "HistogramSnapshot", Record, raising=False)
    health = []
    monkeypatch.setattr(
        telemetry,
        "gauge",
        lambda name, *, unit: Record(set=lambda value, *, attributes: health.append((name, value, attributes))),
    )
    histograms = tuple(
        vllm.VLLMHistogramSnapshot(name=f"family_{index}", buckets=((float("inf"), 1),), count=1, total=1.0, unit="s")
        for index in range(8)
    )
    engines = tuple(
        vllm.VLLMEngineStatsSnapshot(
            engine_id=f"engine-{index}",
            timestamp=1.0,
            current=vllm.VLLMCurrentStats(),
            cumulative=vllm.VLLMCumulativeStats(),
            interval=vllm.VLLMIntervalStats(),
            histograms=histograms,
            histogram_timestamp_ms=1_700_000_000_000,
            histogram_sequence=1,
        )
        for index in range(65)
    )
    published = Record(configured=True, sample_limit_dropped_records=0, telemetry_lost_records=0)
    accepted = []

    def publish_histograms(records):
        accepted.extend(records[:512])
        return Record(
            configured=True,
            sample_limit_dropped_records=max(0, len(records) - 512),
            telemetry_lost_records=0,
        )

    sink = FinelogInferenceMetricsSink.__new__(FinelogInferenceMetricsSink)
    sink._histogram_format = VllmHistogramFormat.STRUCTURED
    sink._publisher = Record(publish=lambda records: published)
    sink._histogram_publisher = Record(publish=publish_histograms)
    sink._bridge_publisher = Record(publish=lambda records: published)

    sink.publish(vllm.InferenceStatsSnapshot(engines), step=1)

    assert {(record.attributes["engine"], record.name) for record in accepted} == {
        (f"engine-{engine}", f"family_{family}") for engine in range(65) for family in range(8)
    }
    assert ("metric_publication_dropped_records", 0, {"metric_source": "vllm", "drop_reason": "sample_limit"}) in health
