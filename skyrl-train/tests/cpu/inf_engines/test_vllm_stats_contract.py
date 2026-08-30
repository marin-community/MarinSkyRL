from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from skyrl_train.inference_engines.vllm.stats import MetricTemporality, collect_native_metrics


def _families():
    registry = CollectorRegistry()
    Gauge("vllm:num_requests_running", "d", ["model"], registry=registry).labels(model="m").set(2)
    Counter("vllm:request_success", "d", ["finished_reason"], registry=registry).labels(
        finished_reason="length"
    ).inc(3)
    Histogram(
        "vllm:e2e_request_latency_seconds", "d", ["model"], buckets=(0.5, 1.0), registry=registry
    ).labels(model="m").observe(0.75)
    Counter("python_gc_collections", "d", registry=registry).inc(99)
    return tuple(registry.collect())


def test_native_producer_preserves_labels_temporality_and_histogram_buckets():
    metrics, histograms = collect_native_metrics(_families())

    assert [(metric.name, metric.value, metric.temporality, metric.attributes) for metric in metrics] == [
        ("num_requests_running", 2.0, MetricTemporality.CURRENT, {"model": "m"}),
        ("request_success_total", 3.0, MetricTemporality.CUMULATIVE, {"finished_reason": "length"}),
    ]
    assert len(histograms) == 1
    assert histograms[0].name == "e2e_request_latency_seconds"
    assert histograms[0].buckets == ((0.5, 0.0), (1.0, 1.0), (float("inf"), 1.0))
    assert histograms[0].count == 1
    assert histograms[0].total == 0.75
    assert histograms[0].attributes == {"model": "m"}


def test_vllm_engine_package_has_no_direct_publisher_or_parallel_prometheus_path():
    package = Path(__file__).parents[3] / "skyrl_train" / "inference_engines" / "vllm"
    forbidden = {
        "enable_ray_prometheus_stats",
        "RayPrometheusStatLogger",
        "MetricSnapshotPublisher",
        "rigging.telemetry",
        "wandb.log",
    }

    findings = {
        (path.relative_to(package), token)
        for path in package.rglob("*.py")
        for token in forbidden
        if token in path.read_text()
    }
    assert findings == set()
