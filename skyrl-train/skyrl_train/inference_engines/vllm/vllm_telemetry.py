"""Forward the engine's own vLLM metrics to Finelog, from inside the engine process.

``enable_ray_prometheus_stats: true`` hands vLLM a ``PrometheusStatLogger`` subclass, which replaces
the stock logger rather than joining it, leaving the registry this reads empty. The engine warns
where the two are combined.
"""

import contextlib
from collections.abc import Iterator

from loguru import logger
from prometheus_client import REGISTRY
from rigging.telemetry.metrics import MetricSnapshotPublisher
from rigging.telemetry.prometheus import PrometheusCollector, prefixed_metric_snapshots

from skyrl_train.telemetry import INFERENCE_ROLE, process_telemetry

# What this forwards is vLLM's own metrics under vLLM's own names, so it publishes under vLLM rather
# than under the process hosting it. marin's serving path uses the same name, so one reader finds both.
VLLM_SERVICE = "vllm"
METRIC_PREFIX = "vllm:"

# One engine is ~153 series. The publisher truncates past the cap silently, keeping the prefix, and
# the timing histograms register last -- so an overflow drops them first. A test holds the set under it.
MAX_SNAPSHOTS = 1024
COLLECTOR_STOP_TIMEOUT = 0.5

# Where rollout time goes, how loaded the engine is, and why a request ended. `cache_config_info` is
# excluded deliberately: its label *names* are configuration keys.
EXPORTED_FAMILIES = frozenset(
    {
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:time_to_first_token_seconds",
        "vllm:request_generation_tokens",
        # Why a request ended: length, stop, abort. The rollout-truncation signal.
        "vllm:request_success",
        "vllm:num_preemptions",
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_queries",
        "vllm:generation_tokens",
        "vllm:prompt_tokens",
        # The levers: how loaded the engine is, and why requests wait. vLLM sets the aggregate to
        # the sum of the two reasons in the same call, so the breakdown costs one series and saves
        # the reader a decomposition.
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
    # is a unix timestamp. Read in-process it arrives as a sample *inside* the counter family, so
    # `prefixed_metric_snapshots` types it cumulative and it forwards as a counter parked at 1.7e9.
    # marin's scrape path needs no such filter: `generate_latest` re-emits `_created` as its own
    # gauge family, so the type survives the HTTP hop.
    return tuple(snapshot for snapshot in snapshots if not snapshot.name.endswith("_created"))


@contextlib.contextmanager
def engine_metrics_telemetry() -> Iterator[None]:
    """Own this engine process's telemetry and forward its vLLM metrics for the block's duration."""
    with process_telemetry(INFERENCE_ROLE, service=VLLM_SERVICE) as owner:
        collector = owner.collector_or_inert(
            PrometheusCollector(
                metric_source=VLLM_SERVICE,
                scraper=_RegistryScraper(),
                processor=exported_snapshots,
                publisher=MetricSnapshotPublisher(
                    max_records=MAX_SNAPSHOTS,
                    attributes={"metric_source": VLLM_SERVICE},
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
