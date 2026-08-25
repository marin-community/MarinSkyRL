"""Forward the engine's own vLLM metrics to Finelog, from inside the engine process.

vLLM registers its full metric set in each engine's ``prometheus_client`` registry whether or not
``enable_ray_prometheus_stats`` is set: it appends its own ``PrometheusStatLogger`` unless it is
handed a ``PrometheusStatLogger`` *subclass*, and this repo hands it a closure. Measured on a live
engine at the pinned fork: 66 ``vllm:*`` families after engine construction, 0 before.

marin forwards the same metrics for its serving path, but by scraping ``/metrics`` over HTTP. That
does not port: SkyRL runs no HTTP surface on the engine (``enable_http_endpoint`` is false by
default and there is no ASGI app), and a ``prometheus_client`` registry is process-global, so the
read has to happen in the process that owns it. Only the processor and publisher halves are shared.
"""

import contextlib
from collections.abc import Iterator

from loguru import logger
from prometheus_client import REGISTRY
from rigging.telemetry.metrics import MetricSnapshotPublisher
from rigging.telemetry.prometheus import PrometheusCollector, prefixed_metric_snapshots

from skyrl_train.telemetry import INFERENCE_ROLE, process_telemetry

METRIC_SOURCE = "vllm"
METRIC_PREFIX = "vllm:"

# Bounds one poll of this process's registry, and the publisher truncates past it silently. The set
# below is ~152 series at a 32k context, and vLLM multiplies every series by the engine cores that
# share a process, so raise this before adding families or widening data parallelism. 1024 matches
# marin's reader for these metrics (`marin/inference/vllm_server.py`); the 512 elsewhere in this repo
# sizes the Ray scraper, an unrelated and smaller set.
MAX_SNAPSHOTS = 1024
COLLECTOR_STOP_TIMEOUT = 0.5

# vLLM registers 66 families. These answer where rollout time goes, how the engine is loaded, and
# why a request ended; the rest are duplicates of these, configuration echoed back, or structurally
# zero. `cache_config_info` is excluded deliberately: it uses configuration keys as label *names*.
EXPORTED_FAMILIES = frozenset(
    {
        # Where a request's time goes. Histograms, so any quantile survives the read.
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:time_to_first_token_seconds",
        # Generation length as a distribution rather than a mean.
        "vllm:request_generation_tokens",
        # Why a request ended: length, stop, abort. The rollout-truncation signal.
        "vllm:request_success",
        # A counter, so a saturation event cannot fall between reads.
        "vllm:num_preemptions",
        # Exact hit rate, as a ratio of two counters.
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_queries",
        # Real throughput, as rates.
        "vllm:generation_tokens",
        "vllm:prompt_tokens",
        # The levers: how loaded the engine is, and why requests wait.
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_waiting_by_reason",
        "vllm:kv_cache_usage_perc",
    }
)


class _RegistryScraper:
    """Read this process's own registry, in place of an HTTP scrape."""

    def scrape(self):
        return tuple(REGISTRY.collect())


def exported_snapshots(families):
    selected = tuple(family for family in families if family.name in EXPORTED_FAMILIES)
    snapshots = prefixed_metric_snapshots(selected, metric_prefix=METRIC_PREFIX)
    # `prometheus_client` emits a `_created` series beside every counter and histogram whose value
    # is a unix timestamp, and `prefixed_metric_snapshots` types it as the cumulative counter it
    # is labelled as. Forwarded, it reads as a counter parked at 1.7e9.
    return tuple(snapshot for snapshot in snapshots if not snapshot.name.endswith("_created"))


@contextlib.contextmanager
def engine_metrics_telemetry() -> Iterator[None]:
    """Own this engine process's telemetry and forward its vLLM metrics for the block's duration."""
    with process_telemetry(INFERENCE_ROLE) as owner:
        collector = owner.collector_or_inert(
            PrometheusCollector(
                metric_source=METRIC_SOURCE,
                scraper=_RegistryScraper(),
                processor=exported_snapshots,
                publisher=MetricSnapshotPublisher(
                    max_records=MAX_SNAPSHOTS,
                    attributes={"metric_source": METRIC_SOURCE},
                ),
            )
        )
        try:
            collector.start()
        except Exception:
            logger.warning("Could not start vLLM metric forwarding; the engine continues", exc_info=True)
        try:
            yield
        finally:
            try:
                collector.stop(timeout=COLLECTOR_STOP_TIMEOUT)
            except Exception:
                logger.warning("Could not stop vLLM metric forwarding", exc_info=True)
