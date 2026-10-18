from dataclasses import replace
from types import SimpleNamespace as Record

import pytest
from skyrl_train.inference_engines.vllm import stats as vllm
from skyrl_train.inference_observability import FinelogInferenceMetricsSink, trainer_metrics


@pytest.mark.parametrize("engine_count", [1, 65])
def test_vllm_stats_reach_finelog(delivered_telemetry, engine_count):
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
    bridge.record_request_outcome("/tokenize", "client_disconnect")
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
    engines = (engine, *(replace(engine, engine_id=f"physical-{index}") for index in range(1, engine_count)))
    snapshot = vllm.InferenceStatsSnapshot(engines, bridge.snapshot(vllm.IntervalReadMode.RESET))

    FinelogInferenceMetricsSink().publish(snapshot, step=7)

    delivered = delivered_telemetry.flush()
    engine_rows = [row for row in delivered if "engine" in row["attributes"]]
    http_rows = [row for row in delivered if row["attributes"].get("metric_source") == "inference_http_bridge"]
    values = {row["name"]: row["value"] for row in engine_rows if row["attributes"].get("engine") == "physical-a"}
    assert values["num_requests_running"] == 2
    assert values["num_requests_waiting"] == 4
    assert values["generation_tokens_total"] == 12
    assert values["prefix_cache_hits_total"] == 5
    assert values["spec_decode_num_drafts_total"] == 10
    assert values["spec_decode_num_draft_tokens_total"] == 30
    assert values["spec_decode_num_accepted_tokens_total"] == 12
    reasons = {
        row["attributes"]["finished_reason"]: row["value"]
        for row in engine_rows
        if row["name"] == "request_success_total" and row["attributes"]["engine"] == "physical-a"
    }
    assert reasons == {"stop": 0, "length": 1, "abort": 0, "error": 0, "repetition": 0}
    assert values["request_time_per_output_token_seconds_sum"] == 0.07
    assert {row["attributes"]["engine"] for row in engine_rows} == {item.engine_id for item in engines}
    assert all("engine" not in row["attributes"] for row in http_rows)
    assert delivered_telemetry.values(
        "metric_publication_dropped_records", metric_source="vllm", drop_reason="sample_limit"
    ) == [0]
    outcomes = [row for row in http_rows if row["name"] == "request_outcome_count"]
    assert [(row["value"], row["attributes"]) for row in outcomes] == [
        (
            1,
            {
                "endpoint": "/tokenize",
                "reason": "client_disconnect",
                "metric_source": "inference_http_bridge",
                "source_kind": "histogram",
                "source_temporality": "cumulative_snapshot",
            },
        )
    ]
    projected = trainer_metrics(snapshot)
    assert projected["vllm/total_finished_requests"] == engine_count
    assert projected["vllm/spec_decode_acceptance_rate"] == 0.4
    assert projected["vllm/spec_decode_mean_acceptance_length"] == 2.2


def test_native_output_length_and_tpot_preserve_disjoint_bins() -> None:
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
    assert {
        histogram.name: (*vllm.explicit_histogram_bins(histogram), histogram.count, histogram.total, histogram.unit)
        for histogram in native.histograms
    } == {
        "request_generation_tokens": ((10.0, 100.0), (2, 1, 0), 3, 120.0, "{token}"),
        "request_time_per_output_token_seconds": ((0.02, 0.1), (2, 0, 0), 2, 0.04, "s"),
    }


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
    assert vllm.explicit_histogram_bins(histogram) == ((0.01, 0.1), (first_bucket, 2, 3))
    assert histogram.count == first_bucket + 5


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
