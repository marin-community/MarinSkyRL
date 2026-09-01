# Canonical vLLM metrics contract

## Context

MarinSkyRL currently has three overlapping vLLM metric paths. `V1LoggingStatLoggerFixed` accumulates a
step window for `InferenceStatsCallback`; vLLM's Prometheus logger accumulates native counters and histograms;
and `enable_ray_prometheus_stats` can replace that logger with `RayPrometheusStatLogger`. The draft
telemetry work in [#460](https://github.com/marin-community/MarinSkyRL/pull/460) routes the callback's
scalar summaries through `trainer.all_metrics`, but separately reads the process-wide Prometheus registry
and publishes it to Finelog. The two outputs overlap while differing in names, aggregation, cadence, and
failure handling.

The callback currently lacks finish-reason counts, waiting reasons, token counters, and complete latency
histograms. A flat tracker payload cannot represent labelled histograms without discarding their labels or
temporality. The implementation therefore needs one collection contract with separate projections for
step trackers and typed telemetry.

## Decision

`InferenceStatsCallback` owns inference metric collection and publication for a training run. No inference-engine
class, stat logger, or background Prometheus collector publishes vLLM metrics directly.

Each inference engine exposes one RPC that returns an immutable `VLLMEngineStatsSnapshot`. The engine's
stat logger is the sole producer. It observes each vLLM scheduler and request-stat update once and maintains
two views of that event stream:

- cumulative counters, histograms, and current gauges for typed telemetry;
- an interval accumulator reset only by a step-boundary read, for exact step summaries.

Periodic reads do not reset the interval accumulator. This lets the callback publish cumulative Finelog
snapshots while generation is still running without changing the next training-step payload. A step-boundary
read returns both views and atomically resets only the interval accumulator.

The wire types are explicit and contain no tracker- or telemetry-specific objects:

```python
@dataclass(frozen=True)
class HistogramSnapshot:
    buckets: tuple[tuple[float, int], ...]
    count: int
    total: float


@dataclass(frozen=True)
class VLLMEngineStatsSnapshot:
    engine_id: str
    observed_at: float
    current: VLLMCurrentStats
    cumulative: VLLMCumulativeStats
    interval: VLLMIntervalStats


class IntervalReadMode(StrEnum):
    PEEK = "peek"
    RESET = "reset"


async def get_vllm_stats_snapshot(mode: IntervalReadMode) -> VLLMEngineStatsSnapshot: ...
```

`VLLMCurrentStats` contains running and waiting requests, waiting requests by reason, and KV-cache usage.
`VLLMCumulativeStats` contains prompt and generation token counts, prefix-cache hits and queries,
preemptions, completions by finish reason, and histograms for queue, prefill, decode, end-to-end and
time-to-first-token latency plus generated tokens per request. `VLLMIntervalStats` contains the current
callback contract: peak and median running and waiting requests, prompt and generation throughput,
KV-cache usage and prefix-cache hit rate; mean and p90 request latencies; finished requests; preemptions;
and observation counts.

Metric names and supported labels are enums or dataclass fields, not free-form dictionaries. Adding a
measurement requires updating the snapshot schema and its projection disposition in the same change.

The callback has two independent sinks:

1. At every configured step boundary, it requests `RESET`, aggregates engine interval views, and adds the flat `vllm/*`
   projection to `trainer.all_metrics`. The trainer's existing tracker owns console, W&B, and other flat
   destinations.
2. A callback-owned periodic task requests `PEEK` and aggregates no engine identities away. It converts the cumulative and
   current views to typed Rigging metric snapshots and sends them through a `VLLMMetricsSink` protocol.
   The Finelog implementation preserves metric names, engine identity, labels, source kind, and
   current-versus-cumulative temporality. It publishes through the trainer's existing `service="marinskyrl"`
   runtime with `metric_source="vllm"`: Rigging owns one runtime per process, so a callback cannot also
   configure an independent `service="vllm"` runtime. Marin's serving dashboards must select either the
   serving service or the trainer service plus metric source, as appropriate.

Both sinks consume `VLLMEngineStatsSnapshot`; neither reads vLLM internals or a Prometheus registry. A
failure in one sink does not suppress the other. Collection failures are reported by the callback and keep
the last successful cumulative cursor; a failed periodic read cannot reset step statistics.

`InferenceEngineClient` transports one snapshot per engine without aggregating or renaming it. Projection
and cross-engine aggregation happen only in the callback, so the RPC layer cannot become a fourth metric
owner.

Remove `enable_ray_prometheus_stats`, `RayPrometheusStatLogger`, the engine-local Prometheus registry
collector, and their configuration and tests. The normal vLLM Prometheus logger is also unnecessary for
MarinSkyRL metrics and must not be installed as a second stat logger. Marin's standalone serving path may
continue scraping `/metrics`; it is a different process boundary and does not participate in this training
contract.

## Enforcement

CPU tests exercise the public producer-to-consumer path:

- feed scheduler and finished-request observations to the real stat logger, read a snapshot, and verify
  current, cumulative, and interval semantics, including that periodic reads do not reset the interval;
- pass snapshots from multiple fake engine RPCs through `InferenceStatsCallback` and verify the flat tracker
  payload and typed recording sink from the same read;
- require every snapshot field to have an explicit tracker, telemetry, or dual disposition and pin the
  externally consumed metric names and labels;
- verify a tracker failure does not suppress typed telemetry, and a typed-sink failure does not corrupt the
  next step interval;
- scan the vLLM inference package's imports and configuration schema to reject direct Rigging publishers,
  Prometheus registry access, `PrometheusStatLogger`, `RayPrometheusStatLogger`, and
  `enable_ray_prometheus_stats` outside the canonical snapshot producer and callback adapter.

The architectural test is a dependency-boundary check, not a claim that tests can prevent deliberately
obfuscated publication code. Repository review remains responsible for changes to the snapshot schema,
callback, and sink protocol.

## Rollout

Implement this contract on #460 after this design is approved. Preserve the 27 existing `vllm/*` tracker
names and the native Finelog names and labels consumed by Marin's vLLM dashboards. Remove the alternate
exporters in the same PR so no release contains both mechanisms.

Run the CPU contract tests and the ordinary MarinSkyRL PR suite. Then run one short multi-engine GPU rollout
with telemetry enabled and verify that step payloads contain all 27 scalar metrics, Finelog receives one
complete engine-labelled histogram set per callback polling interval, finish-reason and waiting-reason labels
survive, and disabling W&B does not change Finelog output. Compare the resulting Finelog series with the
queries introduced by [marin#7863](https://github.com/marin-community/marin/pull/7863), including its service
filter, before merging #460.

The implementation can be reverted as one unit because it does not change training decisions. If the typed
sink fails in production, flat step metrics remain available through the tracker and the callback records the
publication failure.
