import pytest

# An unguarded import of the optional telemetry extra would abort collection for the whole suite
# rather than this module alone.
pytest.importorskip("prometheus_client")
pytest.importorskip("rigging.telemetry")

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from skyrl_train.inference_engines.vllm.vllm_telemetry import (
    COLLECTOR_STOP_TIMEOUT,
    MAX_SNAPSHOTS,
    engine_metrics_telemetry,
    exported_snapshots,
)

# vLLM labels every family with these, one value each per engine process.
ENGINE = {"model_name": "Qwen", "engine": "0"}
# `request_latency_buckets` in vLLM's loggers.py; the histograms are most of the series count.
LATENCY_BUCKETS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0)
FINISH_REASONS = ("stop", "length", "abort", "error", "repetition")
WAITING_REASONS = ("capacity", "deferred")


def engine_families():
    """One engine's registry, labelled the way vLLM labels its own."""
    registry = CollectorRegistry()
    labels = list(ENGINE)

    for name in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
        Gauge(f"vllm:{name}", "d", labels, registry=registry).labels(**ENGINE).set(1)

    waiting = Gauge("vllm:num_requests_waiting_by_reason", "d", labels + ["reason"], registry=registry)
    for reason in WAITING_REASONS:
        waiting.labels(**ENGINE, reason=reason).set(2)

    success = Counter("vllm:request_success", "d", labels + ["finished_reason"], registry=registry)
    for index, reason in enumerate(FINISH_REASONS):
        success.labels(**ENGINE, finished_reason=reason).inc(index)

    for name in ("num_preemptions", "prefix_cache_hits", "prefix_cache_queries", "generation_tokens", "prompt_tokens"):
        Counter(f"vllm:{name}", "d", labels, registry=registry).labels(**ENGINE).inc(7)

    for name in (
        "request_queue_time_seconds",
        "request_prefill_time_seconds",
        "request_decode_time_seconds",
        "e2e_request_latency_seconds",
        "time_to_first_token_seconds",
        "request_generation_tokens",
    ):
        Histogram(f"vllm:{name}", "d", labels, buckets=LATENCY_BUCKETS, registry=registry).labels(**ENGINE).observe(0.4)

    # vLLM reports its cache configuration as a gauge whose *label names* are configuration keys.
    Gauge("vllm:cache_config_info", "d", ["block_size"], registry=registry).labels(block_size="16").set(1)
    # The registry is process-wide, so it also holds whatever else the process registered.
    Counter("python_gc_collections", "d", ["generation"], registry=registry).labels(generation="0").inc(42)

    return tuple(registry.collect())


def test_the_forwarded_series_are_exactly_the_selected_families():
    names = {snapshot.name for snapshot in exported_snapshots(engine_families())}

    assert names == {
        "num_requests_running",
        "num_requests_waiting",
        "num_requests_waiting_by_reason",
        "kv_cache_usage_perc",
        "request_success_total",
        "num_preemptions_total",
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "generation_tokens_total",
        "prompt_tokens_total",
        "request_queue_time_seconds_bucket",
        "request_queue_time_seconds_count",
        "request_queue_time_seconds_sum",
        "request_prefill_time_seconds_bucket",
        "request_prefill_time_seconds_count",
        "request_prefill_time_seconds_sum",
        "request_decode_time_seconds_bucket",
        "request_decode_time_seconds_count",
        "request_decode_time_seconds_sum",
        "e2e_request_latency_seconds_bucket",
        "e2e_request_latency_seconds_count",
        "e2e_request_latency_seconds_sum",
        "time_to_first_token_seconds_bucket",
        "time_to_first_token_seconds_count",
        "time_to_first_token_seconds_sum",
        "request_generation_tokens_bucket",
        "request_generation_tokens_count",
        "request_generation_tokens_sum",
    }


def test_enumerated_labels_stay_attributes_instead_of_becoming_metric_names():
    """`finished_reason` and `reason` are open sets; folding one into a name loses the next value."""
    snapshots = exported_snapshots(engine_families())

    reasons = {
        snapshot.attributes["finished_reason"] for snapshot in snapshots if snapshot.name == "request_success_total"
    }
    assert reasons == set(FINISH_REASONS)

    waiting = {
        snapshot.attributes["reason"] for snapshot in snapshots if snapshot.name == "num_requests_waiting_by_reason"
    }
    assert waiting == set(WAITING_REASONS)


def test_a_creation_timestamp_is_not_forwarded_as_a_counter():
    """`prometheus_client` pairs every counter with a `_created` series holding a unix timestamp."""
    assert not [snapshot for snapshot in exported_snapshots(engine_families()) if snapshot.name.endswith("_created")]


def test_one_engines_families_fit_inside_the_publisher_budget():
    """The publisher truncates past `max_records` by keeping the prefix, and the timing histograms
    register last, so an overflow drops exactly the families this module exists to deliver."""
    assert len(exported_snapshots(engine_families())) <= MAX_SNAPSHOTS


def test_the_engines_rows_are_published_under_vllms_own_service(monkeypatch):
    """These are vLLM's metrics under vLLM's names; marin's serving path publishes them the same way,
    and every reader of them selects on the service."""
    import skyrl_train.telemetry as trainer_telemetry

    monkeypatch.setenv("SKYRL_TELEMETRY_ENDPOINT", "http://finelog.invalid")
    monkeypatch.setenv("SKYRL_RUN_ID", "run-1")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "exec-1")
    configured = {}

    class _Telemetry:
        def configure(self, *, endpoint, service, attributes):
            configured.update(service=service, role=attributes["role"])

        def runtime_status(self):
            return type("S", (), {"configured": False, "queued_records": 0, "lost_records": 0})()

    monkeypatch.setattr(trainer_telemetry, "telemetry", _Telemetry())

    with engine_metrics_telemetry():
        pass

    assert configured == {"service": "vllm", "role": "inference"}


def test_a_configured_engine_starts_and_stops_the_forwarder(monkeypatch):
    """The rest of the suite proves forwarding stays off when it should; this is the other half."""
    import skyrl_train.telemetry as trainer_telemetry

    monkeypatch.setenv("SKYRL_TELEMETRY_ENDPOINT", "http://finelog.invalid")
    monkeypatch.setenv("SKYRL_RUN_ID", "run-1")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "exec-1")
    calls = []

    class _Telemetry:
        def configure(self, **_):
            pass

        def runtime_status(self):
            return type("S", (), {"configured": True, "queued_records": 0, "lost_records": 0})()

        def event(self, *_args, **_kwargs):
            pass

        def shutdown(self, _timeout):
            pass

    class _Collector:
        def start(self):
            calls.append("start")

        def stop(self, *, timeout):
            calls.append(("stop", timeout))

    monkeypatch.setattr(trainer_telemetry, "telemetry", _Telemetry())
    monkeypatch.setattr(
        "skyrl_train.inference_engines.vllm.vllm_telemetry.PrometheusCollector", lambda **_: _Collector()
    )

    with engine_metrics_telemetry():
        assert calls == ["start"]

    assert calls == ["start", ("stop", COLLECTOR_STOP_TIMEOUT)]


def test_an_engine_without_a_telemetry_endpoint_forwards_nothing(monkeypatch):
    monkeypatch.delenv("SKYRL_TELEMETRY_ENDPOINT", raising=False)
    started = []

    class _Collector:
        def start(self):
            started.append("started")

        def stop(self, *, timeout):
            pass

    monkeypatch.setattr(
        "skyrl_train.inference_engines.vllm.vllm_telemetry.PrometheusCollector",
        lambda **_: _Collector(),
    )

    with engine_metrics_telemetry():
        pass

    assert started == []
