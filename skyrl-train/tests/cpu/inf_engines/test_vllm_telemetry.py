import threading

import pytest

# `prometheus_client` and `rigging.telemetry` both come from the optional `telemetry` extra.
# The documented CPU install (`uv sync --frozen --extra dev`) omits it, and an unguarded
# import here aborts collection for the whole suite rather than this module alone.
pytest.importorskip("prometheus_client")
pytest.importorskip("rigging.telemetry")

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from rigging import telemetry

from skyrl_train.inference_engines.vllm.vllm_telemetry import engine_metrics_telemetry, exported_snapshots


def engine_families():
    """One engine's registry, built the way vLLM builds its own."""
    registry = CollectorRegistry()

    Gauge("vllm:num_requests_running", "d", ["model_name"], registry=registry).labels(model_name="Qwen").set(5)
    Gauge("vllm:num_requests_waiting_by_reason", "d", ["reason"], registry=registry).labels(
        reason="kv_cache_exhausted"
    ).set(2)

    success = Counter("vllm:request_success", "d", ["finished_reason"], registry=registry)
    success.labels(finished_reason="length").inc(4)
    success.labels(finished_reason="stop").inc(11)

    latency = Histogram("vllm:e2e_request_latency_seconds", "d", buckets=(1.0, float("inf")), registry=registry)
    latency.observe(0.5)
    latency.observe(2.0)

    # vLLM reports its cache configuration as a gauge whose *label names* are configuration keys.
    Gauge("vllm:cache_config_info", "d", ["block_size"], registry=registry).labels(block_size="16").set(1)
    # The registry is process-wide, so it also holds whatever else the process registered.
    Counter("python_gc_collections", "d", ["generation"], registry=registry).labels(generation="0").inc(42)

    return tuple(registry.collect())


def test_only_the_selected_vllm_families_are_forwarded():
    names = {snapshot.name for snapshot in exported_snapshots(engine_families())}

    assert "num_requests_running" in names
    assert "request_success_total" in names
    assert not [name for name in names if "cache_config" in name or "gc_collections" in name]


def test_enumerated_labels_stay_attributes_instead_of_becoming_metric_names():
    """`finished_reason` and `reason` are open sets; folding one into a name loses the next value."""
    snapshots = exported_snapshots(engine_families())

    reasons = {
        snapshot.attributes["finished_reason"]: snapshot.value
        for snapshot in snapshots
        if snapshot.name == "request_success_total"
    }
    assert reasons == {"length": 4.0, "stop": 11.0}

    waiting = next(snapshot for snapshot in snapshots if snapshot.name == "num_requests_waiting_by_reason")
    assert waiting.attributes == {"reason": "kv_cache_exhausted"}


def test_histogram_buckets_survive_so_a_quantile_can_be_read_downstream():
    buckets = {
        snapshot.attributes["le"]: snapshot.value
        for snapshot in exported_snapshots(engine_families())
        if snapshot.name == "e2e_request_latency_seconds_bucket"
    }

    assert buckets == {"1.0": 1.0, "+Inf": 2.0}


def test_temporality_follows_the_prometheus_type():
    by_name = {snapshot.name: snapshot for snapshot in exported_snapshots(engine_families())}

    assert by_name["num_requests_running"].source_temporality == telemetry.CURRENT_SNAPSHOT
    assert by_name["request_success_total"].source_temporality == telemetry.CUMULATIVE_SNAPSHOT


def test_a_creation_timestamp_is_not_forwarded_as_a_counter():
    """`prometheus_client` pairs every counter with a `_created` series holding a unix timestamp."""
    assert not [snapshot for snapshot in exported_snapshots(engine_families()) if snapshot.name.endswith("_created")]


def test_an_engine_without_a_telemetry_endpoint_forwards_nothing_and_still_runs(monkeypatch):
    monkeypatch.delenv("SKYRL_TELEMETRY_ENDPOINT", raising=False)
    before = {thread.name for thread in threading.enumerate()}

    with engine_metrics_telemetry():
        pass

    assert not [name for name in {thread.name for thread in threading.enumerate()} - before if "vllm" in name]
