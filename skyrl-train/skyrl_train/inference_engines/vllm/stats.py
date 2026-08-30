"""Typed boundary between vLLM metric producers and trainer observability sinks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class IntervalReadMode(StrEnum):
    """Whether an engine read preserves or advances its interval accumulator."""

    PEEK = "peek"
    RESET = "reset"


class MetricTemporality(StrEnum):
    CURRENT = "current"
    CUMULATIVE = "cumulative"


@dataclass(frozen=True)
class VLLMMetricSample:
    """One labelled scalar from vLLM's native metric surface."""

    name: str
    value: float
    unit: str
    kind: str
    temporality: MetricTemporality
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VLLMHistogramSnapshot:
    """One labelled cumulative histogram, including its bucket boundaries."""

    name: str
    buckets: tuple[tuple[float, float], ...]
    count: float
    total: float
    unit: str
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VLLMIntervalStats:
    """Measurements accumulated by one engine since the last reset read."""

    peak_prompt_throughput: float = 0.0
    peak_generation_throughput: float = 0.0
    peak_running_reqs: int = 0
    peak_waiting_reqs: int = 0
    peak_gpu_cache_usage_perc: float = 0.0
    peak_prefix_cache_hit_rate: float = 0.0
    median_prompt_throughput: float = 0.0
    median_generation_throughput: float = 0.0
    median_running_reqs: float = 0.0
    median_waiting_reqs: float = 0.0
    median_gpu_cache_usage_perc: float = 0.0
    median_prefix_cache_hit_rate: float = 0.0
    mean_prompt_throughput: float = 0.0
    mean_generation_throughput: float = 0.0
    latency_prefill_mean: float = 0.0
    latency_prefill_median: float = 0.0
    latency_prefill_p90: float = 0.0
    latency_decode_mean: float = 0.0
    latency_decode_median: float = 0.0
    latency_decode_p90: float = 0.0
    latency_e2e_mean: float = 0.0
    latency_e2e_median: float = 0.0
    latency_e2e_p90: float = 0.0
    latency_queued_mean: float = 0.0
    latency_queued_median: float = 0.0
    latency_queued_p90: float = 0.0
    latency_ttft_mean: float = 0.0
    latency_ttft_median: float = 0.0
    latency_ttft_p90: float = 0.0
    finished_requests: int = 0
    preempted_reqs: int = 0
    samples: int = 0
    active_samples: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> VLLMIntervalStats:
        return cls(
            **{
                target: values.get(source, default)
                for target, source, default in _INTERVAL_FIELDS
            }
        )


_INTERVAL_FIELDS: tuple[tuple[str, str, float | int], ...] = (
    ("peak_prompt_throughput", "peak_prompt_throughput", 0.0),
    ("peak_generation_throughput", "peak_generation_throughput", 0.0),
    ("peak_running_reqs", "peak_running_reqs", 0),
    ("peak_waiting_reqs", "peak_waiting_reqs", 0),
    ("peak_gpu_cache_usage_perc", "peak_gpu_cache_usage_perc", 0.0),
    ("peak_prefix_cache_hit_rate", "peak_prefix_cache_hit_rate", 0.0),
    ("median_prompt_throughput", "median_prompt_throughput", 0.0),
    ("median_generation_throughput", "median_generation_throughput", 0.0),
    ("median_running_reqs", "median_running_reqs", 0.0),
    ("median_waiting_reqs", "median_waiting_reqs", 0.0),
    ("median_gpu_cache_usage_perc", "median_gpu_cache_usage_perc", 0.0),
    ("median_prefix_cache_hit_rate", "median_prefix_cache_hit_rate", 0.0),
    ("mean_prompt_throughput", "mean_prompt_throughput", 0.0),
    ("mean_generation_throughput", "mean_generation_throughput", 0.0),
    ("latency_prefill_mean", "latency_prefill_mean", 0.0),
    ("latency_prefill_median", "latency_prefill_median", 0.0),
    ("latency_prefill_p90", "latency_prefill_p90", 0.0),
    ("latency_decode_mean", "latency_decode_mean", 0.0),
    ("latency_decode_median", "latency_decode_median", 0.0),
    ("latency_decode_p90", "latency_decode_p90", 0.0),
    ("latency_e2e_mean", "latency_e2e_mean", 0.0),
    ("latency_e2e_median", "latency_e2e_median", 0.0),
    ("latency_e2e_p90", "latency_e2e_p90", 0.0),
    ("latency_queued_mean", "latency_queued_mean", 0.0),
    ("latency_queued_median", "latency_queued_median", 0.0),
    ("latency_queued_p90", "latency_queued_p90", 0.0),
    ("latency_ttft_mean", "latency_ttft_mean", 0.0),
    ("latency_ttft_median", "latency_ttft_median", 0.0),
    ("latency_ttft_p90", "latency_ttft_p90", 0.0),
    ("finished_requests", "latency_num_finished_requests", 0),
    ("preempted_reqs", "total_preempted_reqs", 0),
    ("samples", "num_samples", 0),
    ("active_samples", "num_active_samples", 0),
)


@dataclass(frozen=True)
class VLLMEngineStatsSnapshot:
    """Everything the trainer can observe from one vLLM engine in one read."""

    engine_id: str
    timestamp: float
    interval: VLLMIntervalStats
    metrics: tuple[VLLMMetricSample, ...] = ()
    histograms: tuple[VLLMHistogramSnapshot, ...] = ()


@dataclass(frozen=True)
class VLLMStatsSnapshot:
    """An unaggregated, lossless read from all live inference engines."""

    engines: tuple[VLLMEngineStatsSnapshot, ...]


_NATIVE_METRICS = {
    "vllm:num_requests_running": ("{request}", MetricTemporality.CURRENT),
    "vllm:num_requests_waiting": ("{request}", MetricTemporality.CURRENT),
    "vllm:num_requests_waiting_by_reason": ("{request}", MetricTemporality.CURRENT),
    "vllm:kv_cache_usage_perc": ("1", MetricTemporality.CURRENT),
    "vllm:request_success": ("{request}", MetricTemporality.CUMULATIVE),
    "vllm:num_preemptions": ("{request}", MetricTemporality.CUMULATIVE),
    "vllm:prefix_cache_hits": ("{token}", MetricTemporality.CUMULATIVE),
    "vllm:prefix_cache_queries": ("{token}", MetricTemporality.CUMULATIVE),
    "vllm:generation_tokens": ("{token}", MetricTemporality.CUMULATIVE),
    "vllm:prompt_tokens": ("{token}", MetricTemporality.CUMULATIVE),
}
_NATIVE_HISTOGRAMS = {
    "vllm:request_queue_time_seconds": "s",
    "vllm:request_prefill_time_seconds": "s",
    "vllm:request_decode_time_seconds": "s",
    "vllm:e2e_request_latency_seconds": "s",
    "vllm:time_to_first_token_seconds": "s",
    "vllm:request_generation_tokens": "{token}",
}


def collect_native_metrics(families: Iterable[object]) -> tuple[tuple[VLLMMetricSample, ...], tuple[VLLMHistogramSnapshot, ...]]:
    """Normalize vLLM's registry into the sink-neutral snapshot contract."""
    metrics: list[VLLMMetricSample] = []
    histograms: list[VLLMHistogramSnapshot] = []
    for family in families:
        family_name = getattr(family, "name", "")
        if family_name in _NATIVE_METRICS:
            unit, temporality = _NATIVE_METRICS[family_name]
            expected_name = family_name if temporality is MetricTemporality.CURRENT else f"{family_name}_total"
            for sample in getattr(family, "samples", ()):
                if sample.name != expected_name:
                    continue
                metrics.append(
                    VLLMMetricSample(
                        name=sample.name.removeprefix("vllm:"),
                        value=float(sample.value),
                        unit=unit,
                        kind="gauge" if temporality is MetricTemporality.CURRENT else "counter",
                        temporality=temporality,
                        attributes=dict(sample.labels),
                    )
                )
        elif family_name in _NATIVE_HISTOGRAMS:
            grouped: defaultdict[tuple[tuple[str, str], ...], dict[str, object]] = defaultdict(
                lambda: {"buckets": [], "count": 0.0, "total": 0.0}
            )
            for sample in getattr(family, "samples", ()):
                attributes = tuple(sorted((key, value) for key, value in sample.labels.items() if key != "le"))
                group = grouped[attributes]
                if sample.name.endswith("_bucket"):
                    group["buckets"].append((float(sample.labels["le"]), float(sample.value)))
                elif sample.name.endswith("_count"):
                    group["count"] = float(sample.value)
                elif sample.name.endswith("_sum"):
                    group["total"] = float(sample.value)
            for attributes, values in grouped.items():
                histograms.append(
                    VLLMHistogramSnapshot(
                        name=family_name.removeprefix("vllm:"),
                        buckets=tuple(sorted(values["buckets"])),
                        count=values["count"],
                        total=values["total"],
                        unit=_NATIVE_HISTOGRAMS[family_name],
                        attributes=dict(attributes),
                    )
                )
    return tuple(metrics), tuple(histograms)
