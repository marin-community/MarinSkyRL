"""Sink adapters for the canonical inference-service snapshot."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from loguru import logger
from rigging import telemetry as rigging_telemetry
from rigging.telemetry import serialization
from rigging.telemetry.metrics import MetricPublishResult, MetricSnapshot, MetricSnapshotPublisher

try:
    from rigging.telemetry.metrics import (
        CumulativeHistogramBundleSnapshot,
        CumulativeHistogramBundleSnapshotPublisher,
        CumulativeHistogramSnapshot,
        cumulative_histogram_schema,
        cumulative_histogram_series,
    )
except ImportError as error:
    _RIGGING_IMPORT_ERROR = error
    CumulativeHistogramBundleSnapshot = None
    CumulativeHistogramBundleSnapshotPublisher = None
    CumulativeHistogramSnapshot = None
    cumulative_histogram_schema = None
    cumulative_histogram_series = None
else:
    _RIGGING_IMPORT_ERROR = None

from skyrl_train.inference_engines.vllm.stats import (
    HTTP_BRIDGE_METRIC_NAMES,
    VLLM_NUM_ENGINES_METRIC,
    InferenceStatsSnapshot,
    VLLMEngineStatsSnapshot,
    VLLMHistogramSnapshot,
)
from skyrl_train.telemetry import TelemetryConfig


VLLM_MAX_RECORDS_PER_ENGINE = 512
VLLM_MAX_HISTOGRAM_BUNDLES_PER_PUBLICATION = 64
HTTP_BRIDGE_MAX_RECORDS_PER_PUBLICATION = 512
VLLM_HISTOGRAM_BUNDLE_NAME = "vllm_histogram_bundle"
VLLM_METRIC_SOURCE = "vllm"
HTTP_BRIDGE_METRIC_SOURCE = "inference_http_bridge"
METRIC_SOURCE_ATTRIBUTE = "metric_source"
PUBLICATION_LOSS_METRIC = "metric_publication_dropped_records"
ENGINE_ATTRIBUTE = "engine"


@dataclass(frozen=True)
class _PublicationLosses:
    sample_limit: int = 0
    telemetry_loss: int = 0


@dataclass(frozen=True)
class _HistogramPublicationIdentity:
    timestamp_ms: int
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class _HistogramProjection:
    finite_bounds: tuple[float, ...]
    attributes: Mapping[str, str]


class VllmHistogramFormat(StrEnum):
    """Histogram record shapes emitted by the Finelog sink."""

    SCALAR = "scalar"
    STRUCTURED = "structured"
    DUAL = "dual"


class InferenceMetricsSink(Protocol):
    """An interchangeable destination for one callback-owned snapshot."""

    def publish(self, snapshot: InferenceStatsSnapshot, step: int) -> None: ...


def configured_inference_sinks(
    histogram_format: VllmHistogramFormat = VllmHistogramFormat.STRUCTURED,
) -> tuple[InferenceMetricsSink, ...]:
    """Return the Finelog sink when telemetry is configured."""
    if not TelemetryConfig.from_environment().endpoint:
        return ()
    return (FinelogInferenceMetricsSink(histogram_format),)


def trainer_metrics(snapshot: InferenceStatsSnapshot) -> dict[str, float]:
    """Project engine and HTTP bridge observations into the tracker's flat scalar contract."""
    engines = snapshot.engines
    if not engines:
        metrics = {}
    else:
        metrics = _engine_trainer_metrics(engines)
    if snapshot.http_bridge is not None:
        for name in HTTP_BRIDGE_METRIC_NAMES:
            summary = getattr(snapshot.http_bridge, name)
            for statistic in ("count", "mean", "p95", "maximum"):
                metrics[f"inference_bridge/{name}/{statistic}"] = float(getattr(summary, statistic))
    return metrics


def _engine_trainer_metrics(engines: tuple[VLLMEngineStatsSnapshot, ...]) -> dict[str, float]:
    intervals = [engine.interval for engine in engines]
    count = len(intervals)
    finished = sum(item.finished_requests for item in intervals)
    cumulative = [engine.cumulative for engine in engines]
    spec_drafts = sum(item.spec_decode_drafts for item in cumulative)
    spec_draft_tokens = sum(item.spec_decode_draft_tokens for item in cumulative)
    spec_accepted_tokens = sum(item.spec_decode_accepted_tokens for item in cumulative)

    def average(name: str) -> float:
        return sum(float(getattr(item, name)) for item in intervals) / count

    def weighted(name: str) -> float:
        if not finished:
            return 0.0
        return sum(float(getattr(item, name)) * item.finished_requests for item in intervals) / finished

    return {
        VLLM_NUM_ENGINES_METRIC: float(count),
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
        "vllm/spec_decode_drafts_total": float(spec_drafts),
        "vllm/spec_decode_draft_tokens_total": float(spec_draft_tokens),
        "vllm/spec_decode_accepted_tokens_total": float(spec_accepted_tokens),
        "vllm/spec_decode_acceptance_rate": (
            float(spec_accepted_tokens) / spec_draft_tokens if spec_draft_tokens else 0.0
        ),
        "vllm/spec_decode_mean_acceptance_length": (
            1.0 + float(spec_accepted_tokens) / spec_drafts if spec_drafts else 1.0
        ),
    }


def format_console_summary(metrics: Mapping[str, float], step: int) -> str:
    """Format the compact console view without coupling collection to a logger."""
    return (
        f"vLLM Stats (step {step}): engines={metrics[VLLM_NUM_ENGINES_METRIC]:.0f}, "
        f"running={metrics['vllm/median_running_reqs']:.1f}/{metrics['vllm/peak_running_reqs']:.0f}, "
        f"waiting={metrics['vllm/median_waiting_reqs']:.1f}/{metrics['vllm/peak_waiting_reqs']:.0f}, "
        f"generation={metrics['vllm/median_generation_throughput']:.1f}/"
        f"{metrics['vllm/peak_generation_throughput']:.1f} tok/s, "
        f"kv_cache={metrics['vllm/median_gpu_cache_usage_perc']:.1f}/"
        f"{metrics['vllm/peak_gpu_cache_usage_perc']:.1f}%, "
        f"e2e={metrics['vllm/latency_e2e_mean']:.3f}s"
    )


class FinelogInferenceMetricsSink:
    """Convert the neutral snapshot to Rigging records at the publishing edge."""

    def __init__(self, histogram_format: VllmHistogramFormat = VllmHistogramFormat.STRUCTURED) -> None:
        if histogram_format != VllmHistogramFormat.SCALAR and _RIGGING_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The installed Rigging version lacks histogram bundle support"
            ) from _RIGGING_IMPORT_ERROR

        self._histogram_format = histogram_format
        self._publisher = MetricSnapshotPublisher(
            max_records=VLLM_MAX_RECORDS_PER_ENGINE,
            attributes={METRIC_SOURCE_ATTRIBUTE: VLLM_METRIC_SOURCE},
        )
        self._histogram_publisher = (
            CumulativeHistogramBundleSnapshotPublisher(
                max_records=VLLM_MAX_HISTOGRAM_BUNDLES_PER_PUBLICATION,
                attributes={METRIC_SOURCE_ATTRIBUTE: VLLM_METRIC_SOURCE},
            )
            if CumulativeHistogramBundleSnapshotPublisher is not None
            else None
        )
        self._bridge_publisher = MetricSnapshotPublisher(
            max_records=HTTP_BRIDGE_MAX_RECORDS_PER_PUBLICATION,
            attributes={METRIC_SOURCE_ATTRIBUTE: HTTP_BRIDGE_METRIC_SOURCE},
        )

    def publish(self, snapshot: InferenceStatsSnapshot, step: int) -> None:
        losses = self._publish_vllm(snapshot, step)
        _record_publication_health(
            VLLM_METRIC_SOURCE,
            losses.sample_limit,
            losses.telemetry_loss,
        )
        self._publish_http_bridge(snapshot)

    def _publish_vllm(self, snapshot: InferenceStatsSnapshot, step: int) -> _PublicationLosses:
        """Publish atomic per-engine records and return losses by admission stage."""
        sample_limit_dropped = 0
        telemetry_lost = 0
        histogram_bundles: list[CumulativeHistogramBundleSnapshot] = []
        for engine in snapshot.engines:
            engine_attributes = {**engine.attributes, ENGINE_ATTRIBUTE: engine.engine_id}
            publication = _histogram_publication_identity(engine)
            records = _engine_metric_records(
                engine,
                engine_attributes,
                step,
            )
            if self._histogram_format == VllmHistogramFormat.SCALAR:
                records.extend(
                    _engine_scalar_histogram_records(
                        engine,
                        engine_attributes,
                    )
                )
            elif self._histogram_format == VllmHistogramFormat.DUAL:
                records.extend(
                    _engine_correlated_scalar_histogram_records(
                        engine,
                        engine_attributes,
                        publication,
                    )
                )
            losses = self._publish_engine_metrics(engine.engine_id, records)
            sample_limit_dropped += losses.sample_limit
            telemetry_lost += losses.telemetry_loss
            if self._histogram_format in (VllmHistogramFormat.STRUCTURED, VllmHistogramFormat.DUAL):
                bundle = _engine_histogram_bundle(
                    engine,
                    engine_attributes,
                    publication,
                )
                if bundle is not None:
                    histogram_bundles.append(bundle)
        if histogram_bundles:
            if self._histogram_publisher is None:
                raise RuntimeError("Histogram bundles were built without a configured publisher")
            result = self._histogram_publisher.publish(histogram_bundles)
            losses = _publication_losses(result, "vLLM histogram bundle")
            sample_limit_dropped += losses.sample_limit
            telemetry_lost += losses.telemetry_loss
        return _PublicationLosses(sample_limit_dropped, telemetry_lost)

    def _publish_engine_metrics(self, engine_id: str, records: list[MetricSnapshot]) -> _PublicationLosses:
        if len(records) > VLLM_MAX_RECORDS_PER_ENGINE:
            logger.warning(
                "Rejected oversized vLLM metric batch for engine {}: {} records exceeds {}",
                engine_id,
                len(records),
                VLLM_MAX_RECORDS_PER_ENGINE,
            )
            return _PublicationLosses(sample_limit=len(records))
        if not _metric_batch_is_valid(records):
            logger.warning(
                "Rejected invalid vLLM metric batch for engine {} before publication: {} records",
                engine_id,
                len(records),
            )
            return _PublicationLosses(telemetry_loss=len(records))
        if not records:
            return _PublicationLosses()
        return _publication_losses(self._publisher.publish(records), "vLLM metric")

    def _publish_http_bridge(self, snapshot: InferenceStatsSnapshot) -> None:
        if snapshot.http_bridge is not None:
            records = []
            for histogram in snapshot.http_bridge.histograms:
                records.extend(
                    _histogram_records(
                        histogram,
                        {},
                    )
                )
            if records:
                result = self._bridge_publisher.publish(records)
                losses = _publication_losses(result, "HTTP bridge metric")
                _record_publication_health(
                    HTTP_BRIDGE_METRIC_SOURCE,
                    losses.sample_limit,
                    losses.telemetry_loss,
                )


def _engine_metric_records(
    engine: VLLMEngineStatsSnapshot,
    cumulative_base: Mapping[str, str],
    step: int,
) -> list[MetricSnapshot]:
    current_base = {**cumulative_base, "step": str(step)}
    current = engine.current
    records: list[MetricSnapshot] = []
    for name, value, unit, attributes in (
        ("num_requests_running", current.running_requests, "{request}", {}),
        (
            "num_requests_waiting",
            current.waiting_capacity + current.waiting_deferred,
            "{request}",
            {},
        ),
        (
            "num_requests_waiting_by_reason",
            current.waiting_capacity,
            "{request}",
            {"reason": "capacity"},
        ),
        (
            "num_requests_waiting_by_reason",
            current.waiting_deferred,
            "{request}",
            {"reason": "deferred"},
        ),
        ("kv_cache_usage_perc", current.kv_cache_usage, "1", {}),
    ):
        records.append(
            MetricSnapshot(
                name=name,
                value=value,
                unit=unit,
                attributes={**current_base, **attributes},
                source_kind="gauge",
                source_temporality=rigging_telemetry.CURRENT_SNAPSHOT,
            )
        )

    cumulative = engine.cumulative
    counters = (
        ("num_preemptions_total", cumulative.preemptions, "{request}", {}),
        ("prefix_cache_hits_total", cumulative.prefix_cache_hits, "{token}", {}),
        ("prefix_cache_queries_total", cumulative.prefix_cache_queries, "{token}", {}),
        ("generation_tokens_total", cumulative.generation_tokens, "{token}", {}),
        ("prompt_tokens_total", cumulative.prompt_tokens, "{token}", {}),
        ("spec_decode_num_drafts_total", cumulative.spec_decode_drafts, "{draft}", {}),
        ("spec_decode_num_draft_tokens_total", cumulative.spec_decode_draft_tokens, "{token}", {}),
        ("spec_decode_num_accepted_tokens_total", cumulative.spec_decode_accepted_tokens, "{token}", {}),
        *(
            ("request_success_total", value, "{request}", {"finished_reason": reason})
            for reason, value in cumulative.finished_by_reason.items()
        ),
    )
    for name, value, unit, attributes in counters:
        records.append(
            MetricSnapshot(
                name=name,
                value=value,
                unit=unit,
                attributes={**cumulative_base, **attributes},
                source_kind="counter",
                source_temporality=rigging_telemetry.CUMULATIVE_SNAPSHOT,
            )
        )
    return records


def _histogram_publication_identity(engine: VLLMEngineStatsSnapshot) -> _HistogramPublicationIdentity:
    timestamp_ms = int(engine.timestamp * 1_000)
    return _HistogramPublicationIdentity(
        timestamp_ms=timestamp_ms,
        attributes={
            "histogram_collection_timestamp_ms": str(timestamp_ms),
            "histogram_publication_id": f"{engine.engine_id}:{timestamp_ms}:{engine.sample_sequence}",
            "histogram_sample_sequence": str(engine.sample_sequence),
        },
    )


def _engine_scalar_histogram_records(
    engine: VLLMEngineStatsSnapshot,
    cumulative_base: Mapping[str, str],
) -> list[MetricSnapshot]:
    return _engine_histogram_records(engine, cumulative_base, (None,) * len(engine.histograms))


def _engine_correlated_scalar_histogram_records(
    engine: VLLMEngineStatsSnapshot,
    cumulative_base: Mapping[str, str],
    publication: _HistogramPublicationIdentity,
) -> list[MetricSnapshot]:
    additional_attributes = []
    for histogram in engine.histograms:
        projection = _histogram_projection(histogram, cumulative_base)
        additional_attributes.append(
            {
                **publication.attributes,
                "histogram_schema": cumulative_histogram_schema(projection.finite_bounds),
                "histogram_series": cumulative_histogram_series(projection.attributes),
            }
        )
    return _engine_histogram_records(engine, cumulative_base, additional_attributes)


def _engine_histogram_records(
    engine: VLLMEngineStatsSnapshot,
    cumulative_base: Mapping[str, str],
    additional_attributes: Sequence[Mapping[str, str] | None],
) -> list[MetricSnapshot]:
    records = []
    for histogram, attributes in zip(engine.histograms, additional_attributes, strict=True):
        records.extend(_histogram_records(histogram, cumulative_base, additional_attributes=attributes))
    return records


def _engine_histogram_bundle(
    engine: VLLMEngineStatsSnapshot,
    cumulative_base: Mapping[str, str],
    publication: _HistogramPublicationIdentity,
) -> CumulativeHistogramBundleSnapshot | None:
    """Build one exact structured bundle, or None when the engine has no histograms."""
    structured_histograms = []
    for histogram in engine.histograms:
        projection = _histogram_projection(histogram, cumulative_base)
        structured_histograms.append(
            CumulativeHistogramSnapshot(
                name=histogram.name,
                finite_bounds=projection.finite_bounds,
                cumulative_counts=tuple(count for _, count in histogram.buckets),
                count=histogram.count,
                total=histogram.total,
                unit=histogram.unit,
                attributes=projection.attributes,
            )
        )
    if not structured_histograms:
        return None
    return CumulativeHistogramBundleSnapshot(
        name=VLLM_HISTOGRAM_BUNDLE_NAME,
        histograms=tuple(structured_histograms),
        attributes={**cumulative_base, **publication.attributes},
        timestamp_ms=publication.timestamp_ms,
        sample_sequence=engine.sample_sequence,
    )


def _histogram_projection(histogram: VLLMHistogramSnapshot, cumulative_base: Mapping[str, str]) -> _HistogramProjection:
    return _HistogramProjection(
        finite_bounds=tuple(bound for bound, _ in histogram.buckets if math.isfinite(bound)),
        attributes={**histogram.attributes, **cumulative_base},
    )


def _publication_losses(result: MetricPublishResult, label: str) -> _PublicationLosses:
    """Return admitted sample-limit and queue losses; unconfigured publishers report neither."""
    if result.sample_limit_dropped_records or result.telemetry_lost_records:
        logger.warning(
            "{} publication lost records: sample_limit={}, telemetry_queue={}",
            label,
            result.sample_limit_dropped_records,
            result.telemetry_lost_records,
        )
    if not result.configured:
        return _PublicationLosses()
    return _PublicationLosses(result.sample_limit_dropped_records, result.telemetry_lost_records)


def _metric_batch_is_valid(records: list[MetricSnapshot]) -> bool:
    """Mirror Rigging's per-record validation before admitting an engine batch."""
    try:
        for record in records:
            serialization.validate_string(record.name, "name")
            if record.unit:
                serialization.validate_string(record.unit, "unit")
            if not math.isfinite(float(record.value)):
                raise ValueError("metric value must be finite")
            serialization.validate_attributes(
                {
                    **record.attributes,
                    METRIC_SOURCE_ATTRIBUTE: VLLM_METRIC_SOURCE,
                    "source_kind": record.source_kind,
                    "source_temporality": record.source_temporality,
                }
            )
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _histogram_records(
    histogram: VLLMHistogramSnapshot,
    base: Mapping[str, str],
    additional_attributes: Mapping[str, str] | None = None,
) -> list[MetricSnapshot]:
    attributes = {**histogram.attributes, **base}
    if additional_attributes:
        attributes = {**attributes, **additional_attributes}
    records = [
        MetricSnapshot(
            name=f"{histogram.name}_bucket",
            value=value,
            unit=histogram.unit,
            attributes={**attributes, "le": "+Inf" if math.isinf(bound) else str(bound)},
            source_kind="histogram",
            source_temporality=rigging_telemetry.CUMULATIVE_SNAPSHOT,
        )
        for bound, value in histogram.buckets
    ]
    for suffix, value in (("count", histogram.count), ("sum", histogram.total)):
        records.append(
            MetricSnapshot(
                name=f"{histogram.name}_{suffix}",
                value=value,
                unit=histogram.unit,
                attributes=attributes,
                source_kind="histogram",
                source_temporality=rigging_telemetry.CUMULATIVE_SNAPSHOT,
            )
        )
    return records


def _record_publication_health(metric_source: str, sample_limit: int, telemetry_loss: int) -> None:
    """Publish current loss state; this best-effort signal cannot attest to its own delivery."""
    gauge = rigging_telemetry.gauge(PUBLICATION_LOSS_METRIC, unit="{record}")
    for reason, value in (("sample_limit", sample_limit), ("telemetry_loss", telemetry_loss)):
        gauge.set(value, attributes={METRIC_SOURCE_ATTRIBUTE: metric_source, "drop_reason": reason})
