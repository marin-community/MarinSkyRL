from pathlib import Path
from types import SimpleNamespace

from skyrl_train.inference_engines.vllm.stats import VLLMNativeStatsAccumulator, build_1_2_5_buckets
from skyrl_train.inference_engines.vllm.stats import (
    VLLMEngineStatsSnapshot,
    VLLMIntervalStats,
    VLLMStatsSnapshot,
)
from skyrl_train.vllm_observability import FinelogVLLMMetricsSink


def test_native_accumulator_preserves_current_cumulative_labels_and_histograms():
    accumulator = VLLMNativeStatsAccumulator(build_1_2_5_buckets(100), {"model_name": "m", "engine": "0"})
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
    )
    iteration = SimpleNamespace(
        num_prompt_tokens=20,
        num_generation_tokens=12,
        num_preempted_reqs=1,
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
    assert native.cumulative.finished_by_reason == {"length": 1}
    assert {histogram.name for histogram in native.histograms} == {
        "request_queue_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "e2e_request_latency_seconds",
        "time_to_first_token_seconds",
        "request_generation_tokens",
    }
    assert all(histogram.attributes == {"model_name": "m", "engine": "0"} for histogram in native.histograms)
    assert next(item for item in native.histograms if item.name == "request_generation_tokens").total == 12


def test_native_accumulator_is_cumulative_across_reads():
    accumulator = VLLMNativeStatsAccumulator((1, 10), {})
    iteration = SimpleNamespace(
        num_prompt_tokens=2,
        num_generation_tokens=3,
        num_preempted_reqs=0,
        time_to_first_tokens_iter=[],
        finished_requests=[],
    )

    accumulator.observe(None, iteration)
    first = accumulator.snapshot()
    accumulator.observe(None, iteration)
    second = accumulator.snapshot()

    assert first.cumulative.prompt_tokens == 2
    assert second.cumulative.prompt_tokens == 4


def test_finelog_adapter_projects_every_typed_native_measurement():
    accumulator = VLLMNativeStatsAccumulator((1, 10), {"model_name": "m"})
    accumulator.current = accumulator.current.__class__(2, 3, 1, 0.4)
    accumulator.prompt_tokens = 20
    accumulator.generation_tokens = 12
    accumulator.prefix_cache_hits = 5
    accumulator.prefix_cache_queries = 8
    accumulator.preemptions = 1
    accumulator.finished_by_reason = {"length": 2}
    native = accumulator.snapshot()
    snapshot = VLLMStatsSnapshot(
        (
            VLLMEngineStatsSnapshot(
                "engine-a",
                1.0,
                native.current,
                native.cumulative,
                VLLMIntervalStats(),
                attributes={"model_name": "m"},
                histograms=native.histograms,
            ),
        )
    )
    published = []
    sink = FinelogVLLMMetricsSink.__new__(FinelogVLLMMetricsSink)
    sink._publisher = SimpleNamespace(publish=lambda records: published.extend(records))

    sink.publish(snapshot, step=7)

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
    }.issubset(record.name for record in published)
    reasons = {
        record.attributes.get("reason") for record in published if record.name == "num_requests_waiting_by_reason"
    }
    assert reasons == {"capacity", "deferred"}
    assert all(record.attributes["engine"] == "engine-a" for record in published)
    assert all(record.attributes["step"] == "7" for record in published)


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
