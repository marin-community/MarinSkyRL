"""Typed boundary between vLLM metric producers and trainer observability sinks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class IntervalReadMode(StrEnum):
    PEEK = "peek"
    RESET = "reset"


@dataclass(frozen=True)
class VLLMHistogramSnapshot:
    name: str
    buckets: tuple[tuple[float, float], ...]
    count: float
    total: float
    unit: str
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VLLMCurrentStats:
    running_requests: int = 0
    waiting_capacity: int = 0
    waiting_deferred: int = 0
    kv_cache_usage: float = 0.0


@dataclass(frozen=True)
class VLLMCumulativeStats:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
    preemptions: int = 0
    finished_by_reason: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class VLLMIntervalStats:
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
        return cls(**{target: values.get(source, default) for target, source, default in _INTERVAL_FIELDS})


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
    engine_id: str
    timestamp: float
    current: VLLMCurrentStats
    cumulative: VLLMCumulativeStats
    interval: VLLMIntervalStats
    attributes: Mapping[str, str] = field(default_factory=dict)
    histograms: tuple[VLLMHistogramSnapshot, ...] = ()


@dataclass(frozen=True)
class VLLMStatsSnapshot:
    engines: tuple[VLLMEngineStatsSnapshot, ...]


@dataclass
class HistogramAccumulator:
    """Process-local cumulative histogram owned by the canonical stat logger."""

    bounds: tuple[float, ...]
    count: int = 0
    total: float = 0.0
    bucket_counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.bucket_counts:
            self.bucket_counts = [0] * (len(self.bounds) + 1)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for index, bound in enumerate((*self.bounds, math.inf)):
            if value <= bound:
                self.bucket_counts[index] += 1

    def snapshot(self, name: str, unit: str, attributes: Mapping[str, str]) -> VLLMHistogramSnapshot:
        return VLLMHistogramSnapshot(
            name=name,
            buckets=tuple(zip((*self.bounds, math.inf), self.bucket_counts, strict=True)),
            count=self.count,
            total=self.total,
            unit=unit,
            attributes=attributes,
        )


@dataclass
class VLLMNativeStatsAccumulator:
    """Consume vLLM scheduler events once and retain its cumulative typed view."""

    token_bounds: tuple[int, ...]
    attributes: Mapping[str, str]
    current: VLLMCurrentStats = field(default_factory=VLLMCurrentStats)
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_queries: int = 0
    preemptions: int = 0
    finished_by_reason: dict[str, int] = field(default_factory=dict)
    histograms: dict[str, HistogramAccumulator] = field(init=False)

    def __post_init__(self) -> None:
        latency = (
            0.3,
            0.5,
            0.8,
            1.0,
            1.5,
            2.0,
            2.5,
            5.0,
            10.0,
            15.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            120.0,
            240.0,
            480.0,
            960.0,
            1920.0,
            7680.0,
        )
        ttft = (
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
            20.0,
            40.0,
            80.0,
            160.0,
            640.0,
            2560.0,
        )
        self.histograms = {
            "request_queue_time_seconds": HistogramAccumulator(latency),
            "request_prefill_time_seconds": HistogramAccumulator(latency),
            "request_decode_time_seconds": HistogramAccumulator(latency),
            "e2e_request_latency_seconds": HistogramAccumulator(latency),
            "time_to_first_token_seconds": HistogramAccumulator(ttft),
            "request_generation_tokens": HistogramAccumulator(self.token_bounds),
        }

    def observe(self, scheduler_stats: object | None, iteration_stats: object | None) -> None:
        if scheduler_stats is not None:
            self.current = VLLMCurrentStats(
                running_requests=int(scheduler_stats.num_running_reqs),
                waiting_capacity=int(scheduler_stats.num_waiting_reqs),
                waiting_deferred=int(getattr(scheduler_stats, "num_skipped_waiting_reqs", 0)),
                kv_cache_usage=float(scheduler_stats.kv_cache_usage),
            )
            prefix = scheduler_stats.prefix_cache_stats
            self.prefix_cache_hits += int(prefix.hits)
            self.prefix_cache_queries += int(prefix.queries)
        if iteration_stats is None:
            return
        self.prompt_tokens += int(iteration_stats.num_prompt_tokens)
        self.generation_tokens += int(iteration_stats.num_generation_tokens)
        self.preemptions += int(iteration_stats.num_preempted_reqs)
        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.histograms["time_to_first_token_seconds"].observe(float(ttft))
        for request in iteration_stats.finished_requests:
            reason = str(request.finish_reason)
            self.finished_by_reason[reason] = self.finished_by_reason.get(reason, 0) + 1
            for name, value in (
                ("request_queue_time_seconds", request.queued_time),
                ("request_prefill_time_seconds", request.prefill_time),
                ("request_decode_time_seconds", request.decode_time),
                ("e2e_request_latency_seconds", request.e2e_latency),
                ("request_generation_tokens", request.num_generation_tokens),
            ):
                self.histograms[name].observe(float(value))

    def snapshot(self) -> tuple[VLLMCurrentStats, VLLMCumulativeStats, tuple[VLLMHistogramSnapshot, ...]]:
        cumulative = VLLMCumulativeStats(
            prompt_tokens=self.prompt_tokens,
            generation_tokens=self.generation_tokens,
            prefix_cache_hits=self.prefix_cache_hits,
            prefix_cache_queries=self.prefix_cache_queries,
            preemptions=self.preemptions,
            finished_by_reason=dict(self.finished_by_reason),
        )
        histograms = tuple(
            accumulator.snapshot(name, "{token}" if name == "request_generation_tokens" else "s", self.attributes)
            for name, accumulator in self.histograms.items()
        )
        return self.current, cumulative, histograms


def build_1_2_5_buckets(max_value: int) -> tuple[int, ...]:
    buckets = []
    exponent = 0
    while True:
        for mantissa in (1, 2, 5):
            value = mantissa * 10**exponent
            if value > max_value:
                return tuple(buckets)
            buckets.append(value)
        exponent += 1
