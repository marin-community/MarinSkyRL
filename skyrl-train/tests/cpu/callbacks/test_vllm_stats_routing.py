import hashlib
import math
from types import SimpleNamespace as Record

import pytest

from skyrl_train.inference_engines.vllm import stats as vllm
from skyrl_train.inference_observability import (
    FinelogInferenceMetricsSink,
    VllmHistogramFormat,
    trainer_metrics,
)


def _recording_sink(histogram_format):
    scalar_batches = []
    histogram_batches = []
    published = Record(configured=True, sample_limit_dropped_records=0, telemetry_lost_records=0)
    sink = FinelogInferenceMetricsSink(histogram_format)
    sink._publisher = Record(publish=lambda records: scalar_batches.append(records) or published)
    sink._histogram_publisher = Record(publish=lambda records: histogram_batches.append(records) or published)
    sink._bridge_publisher = Record(publish=lambda records: scalar_batches.append(records) or published)
    return Record(sink=sink, scalar_batches=scalar_batches, histogram_batches=histogram_batches)


def test_vllm_stats_reach_finelog():
    labels = {"model_name": "m", "engine": "0"}
    exact_count = (1 << 53) + 1

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
                buckets={"0.01": 0, "0.1": exact_count, "+Inf": exact_count},
                count=exact_count,
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
        9,
    )
    snapshot = vllm.InferenceStatsSnapshot((engine,), bridge.snapshot(vllm.IntervalReadMode.RESET))

    captured = _recording_sink(VllmHistogramFormat.STRUCTURED)
    captured.sink.publish(snapshot, step=7)

    engine, http = captured.scalar_batches
    [bundles] = captured.histogram_batches
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
    assert "request_time_per_output_token_seconds_sum" not in values
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.name == "vllm_histogram_bundle"
    assert bundle.timestamp_ms == 1_000
    assert bundle.sample_sequence == 9
    assert bundle.attributes["engine"] == "physical-a"
    [histogram] = bundle.histograms
    assert histogram.name == "request_time_per_output_token_seconds"
    assert histogram.finite_bounds == (0.01, 0.1)
    assert histogram.cumulative_counts == (0, exact_count, exact_count)
    assert histogram.count == exact_count
    assert histogram.total == 0.07
    assert histogram.attributes == {
        "engine": "physical-a",
        "engine_index": "0",
        "model_name": "m",
    }
    assert all(record.attributes["engine"] == "physical-a" for record in engine)
    assert all("engine" not in record.attributes for record in http)
    projected = trainer_metrics(snapshot)
    assert projected["vllm/total_finished_requests"] == 1
    assert projected["vllm/spec_decode_acceptance_rate"] == 0.4
    assert projected["vllm/spec_decode_mean_acceptance_length"] == 2.2


def test_vllm_histogram_snapshot_rejects_fractional_counts():
    labels = {"model_name": "m", "engine": "0"}
    histogram = Record(
        name="vllm:request_queue_time_seconds",
        labels=labels,
        buckets={"0.1": 0, "+Inf": 1.5},
        count=1.5,
        sum=0.1,
    )

    with pytest.raises(ValueError):
        vllm.snapshot_vllm_prometheus_metrics((histogram,), engine_index="0")


def test_vllm_histogram_bundles_replace_scalar_rows_per_engine():
    bounds = tuple(float(index) for index in range(1, 20)) + (math.inf,)
    histograms = tuple(
        vllm.VLLMHistogramSnapshot(
            name=name,
            buckets=tuple((bound, 0) for bound in bounds),
            count=0,
            total=0.0,
            unit=unit,
            attributes={"engine_index": "0"},
        )
        for name, unit in vllm.VLLM_HISTOGRAM_UNITS.items()
    )
    engines = tuple(
        vllm.VLLMEngineStatsSnapshot(
            engine_id=f"engine-{engine_index}",
            timestamp=1.0,
            current=vllm.VLLMCurrentStats(),
            cumulative=vllm.VLLMCumulativeStats(),
            interval=vllm.VLLMIntervalStats(),
            attributes={"engine_index": str(engine_index)},
            histograms=histograms,
            sample_sequence=3,
        )
        for engine_index in range(32)
    )
    structured = _recording_sink(VllmHistogramFormat.STRUCTURED)
    scalar = _recording_sink(VllmHistogramFormat.SCALAR)

    snapshot = vllm.InferenceStatsSnapshot(engines)
    structured.sink.publish(snapshot, step=7)
    scalar.sink.publish(snapshot, step=7)

    histogram_names = set(vllm.VLLM_HISTOGRAM_UNITS)
    scalar_histogram_rows = sum(
        record.name.rsplit("_", maxsplit=1)[0] in histogram_names for batch in scalar.scalar_batches for record in batch
    )
    structured_bundles = [bundle for batch in structured.histogram_batches for bundle in batch]
    expected_scalar_rows = len(engines) * len(histogram_names) * (len(bounds) + 2)
    assert scalar_histogram_rows == expected_scalar_rows
    assert len(structured_bundles) == len(engines)
    assert all(len(bundle.histograms) == len(histogram_names) for bundle in structured_bundles)


@pytest.mark.parametrize(
    ("histogram_format", "scalar_expected", "structured_expected"),
    (
        (VllmHistogramFormat.SCALAR, True, False),
        (VllmHistogramFormat.DUAL, True, True),
        (VllmHistogramFormat.STRUCTURED, False, True),
    ),
)
def test_vllm_histogram_format_selects_scalar_and_structured_records(
    histogram_format, scalar_expected, structured_expected
):
    histogram = vllm.VLLMHistogramSnapshot(
        name="request_queue_time_seconds",
        buckets=((0.1, 1), (math.inf, 2)),
        count=2,
        total=0.3,
        unit="s",
        attributes={"engine_index": "0"},
    )
    engine = vllm.VLLMEngineStatsSnapshot(
        engine_id="engine-a",
        timestamp=1.0,
        current=vllm.VLLMCurrentStats(),
        cumulative=vllm.VLLMCumulativeStats(),
        interval=vllm.VLLMIntervalStats(),
        attributes={"engine_index": "0"},
        histograms=(histogram,),
        sample_sequence=1,
    )
    captured = _recording_sink(histogram_format)

    captured.sink.publish(vllm.InferenceStatsSnapshot((engine,)), step=1)

    scalar_names = {record.name for batch in captured.scalar_batches for record in batch}
    assert ("request_queue_time_seconds_count" in scalar_names) is scalar_expected
    assert bool(captured.histogram_batches) is structured_expected
    if structured_expected:
        assert len(captured.histogram_batches) == 1
        assert len(captured.histogram_batches[0]) == 1
    if histogram_format == VllmHistogramFormat.DUAL:
        histogram_rows = [
            record
            for batch in captured.scalar_batches
            for record in batch
            if record.name.startswith("request_queue_time_seconds_")
        ]
        [bundle] = captured.histogram_batches[0]
        [structured_histogram] = bundle.histograms
        assert structured_histogram.attributes == {"engine": "engine-a", "engine_index": "0"}
        expected_series = hashlib.sha256(b'{"engine":"engine-a","engine_index":"0"}').hexdigest()
        assert {record.attributes["histogram_series"] for record in histogram_rows} == {expected_series}
