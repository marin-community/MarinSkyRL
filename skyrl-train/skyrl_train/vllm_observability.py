"""Sink adapters for the canonical vLLM engine-stat snapshot."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Protocol

from skyrl_train.inference_engines.vllm.stats import (
    MetricTemporality,
    VLLMStatsSnapshot,
)


class VLLMMetricsSink(Protocol):
    """An interchangeable destination for one callback-owned snapshot."""

    def publish(self, snapshot: VLLMStatsSnapshot, step: int) -> None: ...


def configured_vllm_sinks() -> tuple[VLLMMetricsSink, ...]:
    """Construct optional publishers without exposing their dependencies to the callback."""
    if not os.environ.get("SKYRL_TELEMETRY_ENDPOINT"):
        return ()
    try:
        return (FinelogVLLMMetricsSink(),)
    except ImportError:
        return ()


def trainer_metrics(snapshot: VLLMStatsSnapshot) -> dict[str, float]:
    """Project lossless engine observations into the tracker's flat scalar contract."""
    engines = snapshot.engines
    if not engines:
        return {}
    intervals = [engine.interval for engine in engines]
    count = len(intervals)
    finished = sum(item.finished_requests for item in intervals)

    def average(name: str) -> float:
        return sum(float(getattr(item, name)) for item in intervals) / count

    def weighted(name: str) -> float:
        if not finished:
            return 0.0
        return sum(float(getattr(item, name)) * item.finished_requests for item in intervals) / finished

    return {
        "vllm/num_engines": float(count),
        "vllm/peak_running_reqs": float(sum(item.peak_running_reqs for item in intervals)),
        "vllm/peak_waiting_reqs": float(sum(item.peak_waiting_reqs for item in intervals)),
        "vllm/peak_prompt_throughput": average("peak_prompt_throughput"),
        "vllm/peak_generation_throughput": average("peak_generation_throughput"),
        "vllm/peak_gpu_cache_usage_perc": average("peak_gpu_cache_usage_perc"),
        "vllm/peak_prefix_cache_hit_rate": average("peak_prefix_cache_hit_rate"),
        "vllm/median_running_reqs": average("median_running_reqs"),
        "vllm/median_waiting_reqs": average("median_waiting_reqs"),
        "vllm/median_prompt_throughput": average("median_prompt_throughput"),
        "vllm/median_generation_throughput": average("median_generation_throughput"),
        "vllm/median_gpu_cache_usage_perc": average("median_gpu_cache_usage_perc"),
        "vllm/median_prefix_cache_hit_rate": average("median_prefix_cache_hit_rate"),
        "vllm/latency_prefill_mean": weighted("latency_prefill_mean"),
        "vllm/latency_prefill_p90": max(item.latency_prefill_p90 for item in intervals),
        "vllm/latency_decode_mean": weighted("latency_decode_mean"),
        "vllm/latency_decode_p90": max(item.latency_decode_p90 for item in intervals),
        "vllm/latency_e2e_mean": weighted("latency_e2e_mean"),
        "vllm/latency_e2e_p90": max(item.latency_e2e_p90 for item in intervals),
        "vllm/latency_queued_mean": weighted("latency_queued_mean"),
        "vllm/latency_queued_p90": max(item.latency_queued_p90 for item in intervals),
        "vllm/latency_ttft_mean": weighted("latency_ttft_mean"),
        "vllm/latency_ttft_p90": max(item.latency_ttft_p90 for item in intervals),
        "vllm/total_finished_requests": float(finished),
        "vllm/total_preempted_reqs": float(sum(item.preempted_reqs for item in intervals)),
        "vllm/total_samples": float(sum(item.samples for item in intervals)),
        "vllm/total_active_samples": float(sum(item.active_samples for item in intervals)),
    }


def format_console_summary(metrics: Mapping[str, float], step: int) -> str:
    """Format the compact console view without coupling collection to a logger."""
    return (
        f"vLLM Stats (step {step}): engines={metrics['vllm/num_engines']:.0f}, "
        f"running={metrics['vllm/median_running_reqs']:.1f}/{metrics['vllm/peak_running_reqs']:.0f}, "
        f"waiting={metrics['vllm/median_waiting_reqs']:.1f}/{metrics['vllm/peak_waiting_reqs']:.0f}, "
        f"generation={metrics['vllm/median_generation_throughput']:.1f}/"
        f"{metrics['vllm/peak_generation_throughput']:.1f} tok/s, "
        f"kv_cache={metrics['vllm/median_gpu_cache_usage_perc']:.1f}/"
        f"{metrics['vllm/peak_gpu_cache_usage_perc']:.1f}%, "
        f"e2e={metrics['vllm/latency_e2e_mean']:.3f}s"
    )


class FinelogVLLMMetricsSink:
    """Convert the neutral snapshot to Rigging records at the publishing edge."""

    def __init__(self) -> None:
        from rigging.telemetry.metrics import MetricSnapshotPublisher

        self._publisher = MetricSnapshotPublisher(max_records=512, attributes={"metric_source": "vllm"})

    def publish(self, snapshot: VLLMStatsSnapshot, step: int) -> None:
        from rigging import telemetry
        from rigging.telemetry.metrics import MetricSnapshot

        for engine in snapshot.engines:
            records = []
            base = {"engine": engine.engine_id, "step": str(step)}
            for metric in engine.metrics:
                records.append(
                    MetricSnapshot(
                        name=metric.name,
                        value=metric.value,
                        unit=metric.unit,
                        attributes={**base, **metric.attributes},
                        source_kind=metric.kind,
                        source_temporality=(
                            telemetry.CUMULATIVE_SNAPSHOT
                            if metric.temporality is MetricTemporality.CUMULATIVE
                            else telemetry.CURRENT_SNAPSHOT
                        ),
                    )
                )
            for histogram in engine.histograms:
                records.extend(_histogram_records(histogram, base, MetricSnapshot, telemetry.CUMULATIVE_SNAPSHOT))
            if records:
                self._publisher.publish(records)


def _histogram_records(histogram, base, metric_snapshot_type, cumulative):
    attributes = {**base, **histogram.attributes}
    records = [
        metric_snapshot_type(
            name=f"{histogram.name}_bucket",
            value=value,
            unit=histogram.unit,
            attributes={**attributes, "le": "+Inf" if math.isinf(bound) else str(bound)},
            source_kind="histogram",
            source_temporality=cumulative,
        )
        for bound, value in histogram.buckets
    ]
    for suffix, value in (("count", histogram.count), ("sum", histogram.total)):
        records.append(
            metric_snapshot_type(
                name=f"{histogram.name}_{suffix}",
                value=value,
                unit=histogram.unit,
                attributes=attributes,
                source_kind="histogram",
                source_temporality=cumulative,
            )
        )
    return records
