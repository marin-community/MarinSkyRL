"""The prefix cache hit rate is a share of queried tokens, sampled only on admissions.

Run with:
uv run --frozen pytest tests/cpu/inf_engines/vllm/test_prefix_cache_hit_rate.py
"""

import dataclasses
from dataclasses import dataclass

import pytest

from skyrl_train.inference_engines.vllm.utils import PrefixCacheHitRateAccumulator, prefix_cache_hit_rate_percent


@dataclass
class FakePrefixCacheStats:
    """The field set of vLLM's `PrefixCacheStats`, which carries no miss counter."""

    reset: bool = False
    requests: int = 0
    queries: int = 0
    hits: int = 0
    preempted_requests: int = 0
    preempted_queries: int = 0
    preempted_hits: int = 0


@pytest.mark.parametrize(
    ("hits", "expected"),
    [(1, 0.1), (250, 25.0), (300, 30.0), (500, 50.0), (1000, 100.0)],
)
def test_the_rate_is_the_cached_share_of_queried_tokens(hits, expected):
    """A denominator of `hits` alone collapses every one of these to 100.0."""
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats(requests=1, queries=1000, hits=hits)) == expected


def test_an_iteration_that_queried_nothing_has_no_rate():
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats()) is None


def test_a_genuine_zero_hit_rate_is_still_a_rate():
    """An admission that hit nothing is a real 0.0, distinct from having queried nothing."""
    assert prefix_cache_hit_rate_percent(FakePrefixCacheStats(requests=1, queries=1000, hits=0)) == 0.0


def test_readmitted_preempted_tokens_are_left_out_of_the_rate():
    """vLLM books readmitted preempted requests to a separate set of counters.

    Counting them would report the prefix a preemption forced vLLM to look up a second time as
    if it were a fresh prompt, reading 55.6 here rather than 20.0.
    """
    mixed = FakePrefixCacheStats(
        requests=1, queries=1000, hits=200, preempted_requests=1, preempted_queries=800, preempted_hits=800
    )

    assert prefix_cache_hit_rate_percent(mixed) == 20.0


def test_decode_only_iterations_do_not_drag_the_published_median_down():
    """Most scheduler iterations admit nothing and carry an all-zero delta.

    Sampling 0.0 for each is what published a median of 0.0 beside a peak of 100.0.
    """
    accumulator = PrefixCacheHitRateAccumulator()

    accumulator.observe(FakePrefixCacheStats(requests=1, queries=1000, hits=400), is_active=True)
    for _ in range(9):
        accumulator.observe(FakePrefixCacheStats(), is_active=True)

    assert accumulator.samples == [40.0]
    assert accumulator.peak == 40.0


def test_an_iteration_without_scheduler_stats_is_not_a_zero_hit_rate():
    accumulator = PrefixCacheHitRateAccumulator()

    accumulator.observe(None, is_active=True)

    assert accumulator.samples == []
    assert accumulator.peak == 0.0


def test_the_peak_covers_iterations_the_median_skips():
    """Every metric in this registry peaks over all iterations and samples only active ones."""
    accumulator = PrefixCacheHitRateAccumulator()

    accumulator.observe(FakePrefixCacheStats(requests=1, queries=100, hits=90), is_active=False)

    assert accumulator.peak == 90.0
    assert accumulator.samples == []


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
