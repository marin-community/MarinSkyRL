"""Typed boundary between inference metric producers and trainer observability sinks."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


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
class InferenceStatsSnapshot:
    engines: tuple[VLLMEngineStatsSnapshot, ...]
    http_bridge: HTTPBridgeStatsSnapshot | None = None


@dataclass(frozen=True)
class DistributionSummary:
    count: int = 0
    mean: float = 0.0
    p95: float = 0.0
    maximum: float = 0.0


@dataclass(frozen=True)
class HTTPBridgeStatsSnapshot:
    event_loop_lag_seconds: DistributionSummary = field(default_factory=DistributionSummary)
    response_bytes: DistributionSummary = field(default_factory=DistributionSummary)
    json_serialization_seconds: DistributionSummary = field(default_factory=DistributionSummary)
    histograms: tuple[VLLMHistogramSnapshot, ...] = ()


@dataclass
class _IntervalDistributionAccumulator:
    """Exact count/mean/max with a bounded sample for interval p95."""

    sample_limit: int = 4_096
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0
    samples: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.samples) < self.sample_limit:
            self.samples.append(value)
        else:
            self.samples[(self.count - 1) % self.sample_limit] = value

    def snapshot(self) -> DistributionSummary:
        if not self.count:
            return DistributionSummary()
        ordered = sorted(self.samples)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return DistributionSummary(self.count, self.total / self.count, ordered[p95_index], self.maximum)


class HTTPBridgeStatsAccumulator:
    """Thread-safe bridge observations with PEEK and RESET interval reads."""

    _BOUNDS = {
        "event_loop_lag_seconds": (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
        "response_bytes": (1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576),
        "json_serialization_seconds": (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._interval = {name: _IntervalDistributionAccumulator() for name in self._BOUNDS}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], HistogramAccumulator] = {}

    def observe(self, name: str, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        if name not in self._BOUNDS:
            raise ValueError(f"unknown HTTP bridge metric: {name}")
        labels = tuple(sorted((attributes or {}).items()))
        with self._lock:
            self._interval[name].observe(value)
            histogram = self._histograms.setdefault((name, labels), HistogramAccumulator(self._BOUNDS[name]))
            histogram.observe(value)

    def snapshot(self, read_mode: IntervalReadMode) -> HTTPBridgeStatsSnapshot:
        with self._lock:
            summaries = {name: accumulator.snapshot() for name, accumulator in self._interval.items()}
            histograms = tuple(
                histogram.snapshot(name, "By" if name == "response_bytes" else "s", dict(labels))
                for (name, labels), histogram in self._histograms.items()
            )
            if read_mode is IntervalReadMode.RESET:
                self._interval = {name: _IntervalDistributionAccumulator() for name in self._BOUNDS}
        return HTTPBridgeStatsSnapshot(histograms=histograms, **summaries)


@dataclass(frozen=True)
class VLLMNativeStatsSnapshot:
    current: VLLMCurrentStats
    cumulative: VLLMCumulativeStats
    histograms: tuple[VLLMHistogramSnapshot, ...]


class PrefixCacheStatsLike(Protocol):
    hits: int
    queries: int


class SchedulerStatsLike(Protocol):
    num_running_reqs: int
    num_waiting_reqs: int
    num_skipped_waiting_reqs: int
    kv_cache_usage: float
    prefix_cache_stats: PrefixCacheStatsLike


class FinishedRequestStatsLike(Protocol):
    finish_reason: str
    queued_time: float
    prefill_time: float
    decode_time: float
    e2e_latency: float
    num_generation_tokens: int


class IterationStatsLike(Protocol):
    num_prompt_tokens: int
    num_generation_tokens: int
    num_preempted_reqs: int
    time_to_first_tokens_iter: Sequence[float]
    finished_requests: Sequence[FinishedRequestStatsLike]


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

    def observe(self, scheduler_stats: SchedulerStatsLike | None, iteration_stats: IterationStatsLike | None) -> None:
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

    def snapshot(self) -> VLLMNativeStatsSnapshot:
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
        return VLLMNativeStatsSnapshot(self.current, cumulative, histograms)


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
