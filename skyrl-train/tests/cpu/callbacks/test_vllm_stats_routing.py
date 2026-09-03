from types import SimpleNamespace as Record

from skyrl_train.inference_engines.vllm import stats as vllm
from skyrl_train.inference_observability import FinelogInferenceMetricsSink, trainer_metrics


def test_vllm_stats_reach_finelog():
    native = vllm.VLLMNativeStatsAccumulator((1, 10, 100), {"model_name": "m", "engine_index": "0"})
    native.observe(
        Record(
            num_running_reqs=2,
            num_waiting_reqs=3,
            num_skipped_waiting_reqs=1,
            kv_cache_usage=0.4,
            prefix_cache_stats=Record(hits=5, queries=8),
        ),
        Record(
            num_prompt_tokens=20,
            num_generation_tokens=12,
            num_preempted_reqs=1,
            prompt_token_stats=Record(computed=7),
            time_to_first_tokens_iter=[0.3],
            finished_requests=[
                Record(
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
    bridge = vllm.HTTPBridgeStatsAccumulator()
    bridge.observe("response_bytes", 100, attributes={"status": "2xx"})
    stats = native.snapshot()
    interval = vllm.VLLMIntervalStats(finished_requests=1)
    engine = vllm.VLLMEngineStatsSnapshot(
        "physical-a", 1.0, stats.current, stats.cumulative, interval, {}, stats.histograms
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
    assert values["generation_tokens_total"] == 12
    assert any(r.name == "request_success_total" and r.value == 1 for r in engine)
    assert values["request_time_per_output_token_seconds_sum"] == 0.07
    assert all(record.attributes["engine"] == "physical-a" for record in engine)
    assert all("engine" not in record.attributes for record in http)
    assert trainer_metrics(snapshot)["vllm/total_finished_requests"] == 1
