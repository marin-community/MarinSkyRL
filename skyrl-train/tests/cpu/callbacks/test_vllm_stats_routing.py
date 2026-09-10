from types import SimpleNamespace as Record

from skyrl_train.inference_engines.vllm import stats as vllm
from skyrl_train.inference_observability import FinelogInferenceMetricsSink, trainer_metrics


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
            metric("request_success", 1, finished_reason="length"),
            Record(
                name="vllm:request_time_per_output_token_seconds",
                labels=labels,
                buckets={"0.01": 0, "0.1": 1, "+Inf": 1},
                count=1,
                sum=0.07,
            ),
            Record(
                name="vllm:inter_token_latency_seconds",
                labels=labels,
                buckets={"0.01": 1, "0.025": 9, "+Inf": 12},
                count=12,
                sum=0.35,
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
    sink._publisher = Record(publish=lambda records: batches.append(records) or published)
    sink._bridge_publisher = Record(publish=lambda records: batches.append(records) or published)
    sink.publish(snapshot, step=7)

    engine, http = batches
    values = {record.name: record.value for record in engine}
    assert values["num_requests_running"] == 2
    assert values["num_requests_waiting"] == 4
    assert values["generation_tokens_total"] == 12
    assert values["prefix_cache_hits_total"] == 5
    reasons = {r.attributes["finished_reason"]: r.value for r in engine if r.name == "request_success_total"}
    assert reasons == {"stop": 0, "length": 1, "abort": 0, "error": 0, "repetition": 0}
    assert values["request_time_per_output_token_seconds_sum"] == 0.07
    itl = [record for record in engine if record.name.startswith("inter_token_latency_seconds_")]
    assert {record.attributes["le"]: record.value for record in itl if record.name.endswith("_bucket")} == {
        "0.01": 1,
        "0.025": 9,
        "+Inf": 12,
    }
    assert values["inter_token_latency_seconds_count"] == 12
    assert values["inter_token_latency_seconds_sum"] == 0.35
    assert all(record.attributes["engine_index"] == "0" for record in itl)
    assert all("step" not in record.attributes for record in itl)
    assert all(record.attributes["engine"] == "physical-a" for record in engine)
    assert all("engine" not in record.attributes for record in http)
    assert trainer_metrics(snapshot)["vllm/total_finished_requests"] == 1
