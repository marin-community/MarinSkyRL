"""``record_engine_stats`` exports one step's engine reductions to the telemetry runtime.

The aggregate is exported on every non-empty payload; per-engine series only past the first engine.
A key the payload does not carry is skipped, and a reported zero is exported.
"""

from unittest.mock import patch

import pytest

from skyrl_train import telemetry

AGGREGATE = {
    "num_engines": 1,
    "total_peak_running_reqs": 12,
    "avg_median_running_reqs": 7.5,
    "avg_peak_generation_throughput": 1450.5,
    "avg_latency_e2e_mean": 4.25,
    "max_latency_e2e_p90": 9.5,
    "total_finished_requests": 64,
    "total_preempted_reqs": 3,
}

TWO_ENGINES = {
    **AGGREGATE,
    "num_engines": 2,
    "engines": [
        {"peak_running_reqs": 5, "latency_e2e_mean": 4.0, "latency_num_finished_requests": 30},
        {"peak_running_reqs": 7, "latency_e2e_mean": 4.5, "latency_num_finished_requests": 34},
    ],
}


@pytest.fixture
def recorded():
    """Every ``(name, value, attributes)`` the exporter would receive."""
    calls = []

    def capture(name):
        def record(value, *, attributes=None):
            calls.append((name, value, dict(attributes or {})))

        return record

    instruments = {
        "num_requests_running": telemetry.requests_running,
        "num_requests_waiting": telemetry.requests_waiting,
        "gpu_cache_usage_perc": telemetry.gpu_cache_usage,
        "prefix_cache_hit_rate": telemetry.prefix_cache_hit_rate,
        "prompt_throughput_tokens_per_second": telemetry.prompt_throughput,
        "generation_throughput_tokens_per_second": telemetry.generation_throughput,
        "request_latency_seconds": telemetry.request_latency,
    }
    counters = {
        "requests_finished": telemetry.requests_finished,
        "num_preemptions_total": telemetry.requests_preempted,
    }
    patches = [patch.object(instrument, "set", capture(name)) for name, instrument in instruments.items()]
    patches += [patch.object(instrument, "add", capture(name)) for name, instrument in counters.items()]
    for active in patches:
        active.start()
    try:
        yield calls
    finally:
        for active in patches:
            active.stop()


def _find(calls, name, **attributes):
    return [
        value
        for call_name, value, call_attributes in calls
        if call_name == name and all(call_attributes.get(k) == v for k, v in attributes.items())
    ]


def test_aggregate_reductions_are_exported_under_vllm_metric_names(recorded):
    telemetry.record_engine_stats(AGGREGATE)

    assert _find(recorded, "num_requests_running", engine="all", statistic="peak") == [12.0]
    assert _find(recorded, "num_requests_running", engine="all", statistic="median") == [7.5]
    assert _find(recorded, "generation_throughput_tokens_per_second", engine="all", statistic="peak") == [1450.5]
    assert _find(recorded, "request_latency_seconds", engine="all", stage="e2e", statistic="mean") == [4.25]
    assert _find(recorded, "request_latency_seconds", engine="all", stage="e2e", statistic="p90") == [9.5]


def test_step_totals_are_added_as_counter_deltas(recorded):
    """get_stats resets the accumulators as it reads them, so each count is that step's delta."""
    telemetry.record_engine_stats(AGGREGATE)

    assert _find(recorded, "requests_finished", engine="all") == [64.0]
    assert _find(recorded, "num_preemptions_total", engine="all") == [3.0]


def test_every_series_carries_the_inference_role(recorded):
    telemetry.record_engine_stats(TWO_ENGINES)

    assert recorded
    assert all(attributes["role"] == "inference" for _, _, attributes in recorded)


def test_per_engine_series_appear_only_with_more_than_one_engine(recorded):
    telemetry.record_engine_stats(AGGREGATE)
    assert _find(recorded, "num_requests_running", engine="0") == []

    recorded.clear()
    telemetry.record_engine_stats(TWO_ENGINES)
    assert _find(recorded, "num_requests_running", engine="0", statistic="peak") == [5.0]
    assert _find(recorded, "num_requests_running", engine="1", statistic="peak") == [7.0]
    assert _find(recorded, "requests_finished", engine="1") == [34.0]


def test_an_engineless_payload_exports_nothing(recorded):
    telemetry.record_engine_stats({"num_engines": 0})

    assert recorded == []


def test_a_missing_key_is_skipped_rather_than_exported_as_zero(recorded):
    """A zero the engine never reported would read as a real measurement on a dashboard."""
    telemetry.record_engine_stats({"num_engines": 1, "total_peak_running_reqs": 1})

    assert _find(recorded, "num_requests_waiting") == []
    assert _find(recorded, "request_latency_seconds") == []
    assert _find(recorded, "requests_finished") == []


def test_a_reported_zero_is_exported_rather_than_skipped(recorded):
    """A step that finished no requests is a measurement; a gap in the series is not."""
    telemetry.record_engine_stats({"num_engines": 1, "total_finished_requests": 0})

    assert _find(recorded, "requests_finished", engine="all") == [0.0]
