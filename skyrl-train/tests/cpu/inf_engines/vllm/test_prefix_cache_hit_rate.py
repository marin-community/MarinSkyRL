"""The prefix cache hit rate is a share of queried tokens, not a share of requests.

Run with:
uv run --frozen pytest tests/cpu/inf_engines/vllm/test_prefix_cache_hit_rate.py
"""

import dataclasses
from dataclasses import dataclass

import pytest

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


def test_an_iteration_that_queried_nothing_contributes_no_sample():
    """Scheduler iterations that admit no new request carry an all-zero delta.

    Reporting 0.0 for one is not a low hit rate, it is no measurement, and decode-only
    iterations outnumber admissions badly enough to pin the median there. vLLM's
    `CachingMetrics.observe` returns early on the same empty delta.
    """
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats()) is None


def test_a_genuine_zero_hit_rate_is_still_a_sample():
    """An admission that hit nothing is a real 0.0, distinct from having queried nothing."""
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats(requests=1, queries=1000, hits=0)) == 0.0


def test_only_preempted_lookups_contribute_no_sample():
    """vLLM books readmitted preempted requests to a separate set of counters.

    Reading those as admission traffic would report the prefix a preemption forced vLLM to
    look up again as if it were a fresh prompt.
    """
    preempted_only = FakePrefixCacheStats(preempted_requests=1, preempted_queries=800, preempted_hits=800)

    assert prefix_cache_hit_rate_percent(preempted_only) is None


def test_the_fake_carries_the_same_fields_as_vllms_prefix_cache_stats():
    """A wrong field name caused this bug, and a hand-written fake cannot catch that alone.

    Skipped in CPU CI, which has no vLLM. It runs wherever the vllm extra is installed and
    fails on the next vLLM bump that renames or drops one of these counters.
    """
    pytest.importorskip("vllm")
    from vllm.v1.metrics.stats import PrefixCacheStats

    assert {f.name for f in dataclasses.fields(FakePrefixCacheStats)} == {
        f.name for f in dataclasses.fields(PrefixCacheStats)
    }
