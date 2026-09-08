"""Audit age-conditioned ratios against hand calculations and pooled tokens."""

import math

import pytest
import torch

from skyrl_train.utils.importance_ratio_diagnostics import (
    LogRatioMonitor,
    mismatch_ratio_metrics,
    ratio_statistics,
)


def test_ratio_statistics_match_hand_values_and_clamp_only_exponentials():
    values = [0, math.log(2), -math.log(2), math.log(4), math.log(1e-6), 0]
    result = ratio_statistics(torch.tensor(values, dtype=torch.float64))
    assert result["frac_outside_0.5_2"] == pytest.approx(2 / 6)
    assert result["frac_below_1e-5"] == pytest.approx(1 / 6)
    assert result["kl_k1"] == pytest.approx(-sum(values) / 6, abs=1e-9)
    assert result["kl_k3"] == pytest.approx(sum(math.exp(x) - x - 1 for x in values) / 6, abs=1e-9)
    assert result["chi2"] == pytest.approx(sum(math.exp(2 * x) for x in values) / 6 - 1, abs=1e-9)
    extreme = ratio_statistics(torch.tensor([1000.0], dtype=torch.float64))
    assert extreme["abs_log_ratio_mean"] == 1000
    assert extreme["mean_squared_log_ratio"] == 1e6
    assert extreme["ess_fraction"] == 1
    assert all(math.isfinite(value) for value in extreme.values())


def test_mismatch_metrics_bucket_by_age_and_absolute_position():
    lengths = torch.tensor([300, 600, 100, 700])
    mask = torch.arange(700).unsqueeze(0) < lengths.unsqueeze(1)
    delta = torch.tensor([0.1, 0.2, 0.3, 0.9], dtype=torch.float64).unsqueeze(1).expand(4, 700)
    rollout = torch.zeros_like(delta)
    rollout[~mask] = math.nan
    result = mismatch_ratio_metrics(delta, rollout, mask, torch.tensor([0, 0, 3, 9]))
    assert result["policy/mismatch/age0/abs_log_ratio_mean"] == pytest.approx((300 * 0.1 + 600 * 0.2) / 900)
    assert result["policy/mismatch/age8+/abs_log_ratio_mean"] == pytest.approx(0.9)
    assert result["policy/mismatch/age0/pos_last256/selected_tokens"] == 512
    assert result["policy/mismatch/age3/pos_first256/selected_tokens"] == 100
    assert result["policy/mismatch/age3/pos_last256/selected_tokens"] == 100
    assert result["policy/mismatch/age1/selected_tokens"] == 0
    assert "policy/mismatch/age1/ess_fraction" not in result
    # Put a single mismatch exactly at the 600-token row's last-window boundary.
    changed = torch.zeros_like(delta)
    changed[1, 343:345] = torch.tensor([10.0, 20.0])
    boundary = mismatch_ratio_metrics(changed, rollout, mask, torch.tensor([0, 0, 3, 9]))
    assert boundary["policy/mismatch/age0/pos_last256/abs_log_ratio_mean"] == pytest.approx(20 / 512)


def test_worker_accumulator_matches_pooled_ess_and_tail_under_unequal_microbatches():
    values = torch.linspace(-10, 10, 3000, dtype=torch.float64)
    monitor = LogRatioMonitor(torch.device("cpu"))
    for shard in (values[:100], values[100:1200], values[1200:]):
        shard = shard.unsqueeze(0)
        monitor.add(shard, torch.zeros_like(shard), torch.ones_like(shard))
    actual, expected = monitor.metrics(), ratio_statistics(values)
    assert actual["log_ratio_ess_fraction"] == pytest.approx(expected["ess_fraction"], abs=1e-10)
    assert actual["log_ratio_abs_p999"] == pytest.approx(expected["abs_log_ratio_p999"], abs=1e-5)
    assert actual["log_ratio_kl_k1"] == pytest.approx(expected["kl_k1"], abs=1e-10)
    assert actual["log_ratio_kl_k3"] == pytest.approx(expected["kl_k3"], rel=1e-10)
    assert actual["log_ratio_p999_valid"] == 1
    empty = LogRatioMonitor(torch.device("cpu")).metrics()
    assert set(actual) == set(empty)
    assert empty["log_ratio_statistics_valid"] == 0
    assert all(math.isfinite(value) for value in empty.values())


def test_worker_tail_matches_concatenation_when_one_small_microbatch_has_every_outlier():
    # The first microbatch has all 100 outliers, while the other 19,900 tokens
    # are zero. Keeping only each batch's top 0.1% loses 99 of the outliers.
    shards = (torch.arange(1, 101, dtype=torch.float64), torch.zeros(19_900, dtype=torch.float64))
    expected = torch.quantile(torch.cat(shards).abs(), 0.999).item()
    local_tail_min = min(torch.topk(shard.abs(), max(1, shard.numel() // 1000)).values.min() for shard in shards)
    assert expected == pytest.approx(80.001)
    assert local_tail_min == 0
    monitor = LogRatioMonitor(torch.device("cpu"))
    for shard in shards:
        shard = shard.unsqueeze(0)
        monitor.add(shard, torch.zeros_like(shard), torch.ones_like(shard))
    actual = monitor.metrics()
    assert actual["log_ratio_p999_valid"] == 1
    assert actual["log_ratio_abs_p999"] == pytest.approx(expected, abs=1e-5)


def test_worker_masks_padding_before_subtraction_and_keeps_unclipped_absolute_delta():
    monitor = LogRatioMonitor(torch.device("cpu"))
    monitor.add(torch.tensor([[1000.0, math.nan]]), torch.tensor([[0.0, math.nan]]), torch.tensor([[1, 0]]))
    actual = monitor.metrics()
    assert actual["log_ratio_abs_mean"] == 1000
    assert actual["log_ratio_mean_squared"] == 1e6
    assert actual["log_ratio_statistics_valid"] == 1
