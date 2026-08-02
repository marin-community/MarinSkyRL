"""The prefix cache hit rate is a share of queried tokens, not a share of requests.

Run with:
uv run --frozen pytest tests/cpu/inf_engines/vllm/test_prefix_cache_hit_rate.py
"""

from dataclasses import dataclass

from skyrl_train.inference_engines.vllm.utils import prefix_cache_hit_rate_percent


@dataclass
class FakePrefixCacheStats:
    """The field set of vLLM's `PrefixCacheStats`, which carries no miss counter.

    Deriving a rate from a field this class does not have is what made the published
    metric constant, so the fake mirrors the real shape exactly. A mock would grow a
    `misses` attribute on demand and hide the defect under test.
    """

    reset: bool = False
    requests: int = 0
    queries: int = 0
    hits: int = 0
    preempted_requests: int = 0
    preempted_queries: int = 0
    preempted_hits: int = 0


def test_partial_hit_rate_is_the_cached_share_of_queried_tokens():
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats(requests=2, queries=1000, hits=300)) == 30.0


def test_rates_between_the_extremes_are_distinguishable():
    """A denominator of `hits` alone collapses every one of these to 100.0."""
    rates = [
        prefix_cache_hit_rate_percent(FakePrefixCacheStats(requests=1, queries=1000, hits=hits))
        for hits in (1, 250, 500, 1000)
    ]

    assert rates == [0.1, 25.0, 50.0, 100.0]


def test_a_fully_cached_prefix_reports_one_hundred_percent():
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats(requests=1, queries=512, hits=512)) == 100.0


def test_an_iteration_that_queried_nothing_reports_zero():
    """Scheduler iterations that admit no new request carry an all-zero delta."""
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats()) == 0.0


def test_readmitted_preempted_requests_do_not_contribute_a_rate():
    """vLLM books preempted lookups to a separate set of counters and excludes them here."""
    preempted_only = FakePrefixCacheStats(preempted_requests=1, preempted_queries=800, preempted_hits=800)

    assert prefix_cache_hit_rate_percent(preempted_only) == 0.0
