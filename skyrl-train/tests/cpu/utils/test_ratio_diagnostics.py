"""Audit age-conditioned ratios against hand calculations and pooled tokens."""

import math
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

from skyrl_train.utils.importance_ratio_diagnostics import (
    LogRatioMonitor,
    mismatch_ratio_metrics,
    ratio_statistics,
    gather_ratio_tensor,
    sum_ratio_tensor,
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


def test_rank_reduction_pools_unequal_token_counts_and_excludes_replicas():
    shards = [torch.arange(1, 101, dtype=torch.float64) / 10, torch.zeros(19_900, dtype=torch.float64)]
    monitors = []
    for values in [*shards, shards[0]]:
        monitor = LogRatioMonitor(torch.device("cpu"))
        values = values.unsqueeze(0)
        monitor.add(values, torch.zeros_like(values), torch.ones_like(values))
        monitors.append(monitor)
    expected = ratio_statistics(torch.cat(shards))
    rank_mean_ess = sum(monitor.metrics()["log_ratio_ess_fraction"] for monitor in monitors[:2]) / 2
    assert abs(rank_mean_ess - expected["ess_fraction"]) > 0.1

    def pooled_monitors():
        barrier = Barrier(len(monitors))
        inputs = [None] * len(monitors)

        def run(rank):
            def gather(tensor):
                inputs[rank] = tensor.clone()
                barrier.wait(timeout=20)
                results = [value.clone() for value in inputs]
                barrier.wait(timeout=20)
                return results

            return monitors[rank].metrics(gather_fn=gather, owns_tokens=rank != 2)

        with ThreadPoolExecutor(max_workers=len(monitors)) as executor:
            results = list(executor.map(run, range(len(monitors))))
        assert all(result == results[0] for result in results)
        return results[0]

    actual = pooled_monitors()
    assert actual["stale/abs_log_ratio_p50"] == 0
    assert actual["stale/abs_log_ratio_p95"] == 0
    assert actual["stale/finite_fraction"] == 1
    assert actual["log_ratio_selected_tokens"] == 20_000
    assert actual["log_ratio_ess_fraction"] == pytest.approx(expected["ess_fraction"], rel=1e-10)
    assert actual["log_ratio_mean"] == pytest.approx(expected["log_ratio_mean"], abs=1e-10)
    assert actual["log_ratio_abs_p999"] == pytest.approx(expected["abs_log_ratio_p999"], abs=1e-5)
    assert actual["log_ratio_p999_valid"] == 1

    # One failing rank invalidates the family on all ranks, rather than being
    # averaged into a fractional validity flag by the later WORLD status mean.
    monitors[1]._failed = True
    failed = pooled_monitors()
    assert failed["log_ratio_diagnostics_failed"] == 1
    assert failed["log_ratio_p999_valid"] == 0
    assert set(failed) == set(actual)


def _distributed_ratio_worker(rank, directory):
    torch.distributed.init_process_group("gloo", init_method=f"file://{directory}/group", rank=rank, world_size=2)
    try:
        values = torch.arange(1, 101, dtype=torch.float64) / 10 if rank == 0 else torch.zeros(19_900)
        monitor = LogRatioMonitor(torch.device("cpu"))
        values = values.unsqueeze(0)
        monitor.add(values, torch.zeros_like(values), torch.ones_like(values))
        pooled = monitor.metrics(gather_fn=gather_ratio_tensor, sum_reduce_fn=sum_ratio_tensor)
        if rank == 1:
            monitor._failed = True
        failed = monitor.metrics(gather_fn=gather_ratio_tensor, sum_reduce_fn=sum_ratio_tensor)
        Path(directory, f"rank{rank}.json").write_text(json.dumps({"pooled": pooled, "failed": failed}))
    finally:
        torch.distributed.destroy_process_group()


def test_two_actual_gloo_ranks_emit_identical_token_pooled_statistics(tmp_path):
    torch.multiprocessing.spawn(_distributed_ratio_worker, args=(str(tmp_path),), nprocs=2, join=True)
    left, right = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert left == right
    expected = ratio_statistics(torch.cat([torch.arange(1, 101, dtype=torch.float64) / 10, torch.zeros(19_900)]))
    assert left["pooled"]["log_ratio_selected_tokens"] == 20_000
    assert left["pooled"]["log_ratio_ess_fraction"] == pytest.approx(expected["ess_fraction"], rel=1e-10)
    assert left["pooled"]["log_ratio_abs_p999"] == pytest.approx(expected["abs_log_ratio_p999"], abs=1e-5)
    assert left["pooled"]["log_ratio_p999_valid"] == 1
    assert left["failed"]["log_ratio_p999_valid"] == 0
    assert left["failed"]["log_ratio_diagnostics_failed"] == 1


def test_worker_full_statistic_coverage_clip_bounds_and_nonfinite_counts():
    values = torch.tensor([[-2.0, -0.25, 0.0, 0.5, 3.0]], dtype=torch.float64)
    monitor = LogRatioMonitor(torch.device("cpu"), eps_clip_low=0.1, eps_clip_high=0.3)
    monitor.add(values, torch.zeros_like(values), torch.ones_like(values))
    actual = monitor.metrics()
    oracle = ratio_statistics(values, eps_clip_low=0.1, eps_clip_high=0.3)
    for key in (
        "abs_log_ratio_p50",
        "abs_log_ratio_p95",
        "finite_fraction",
        "lower_clip_pressure",
        "upper_clip_pressure",
    ):
        assert actual[f"stale/{key}"] == pytest.approx(oracle[key], abs=1e-12)
    assert actual["stale/quantiles_valid"] == 1
    assert actual["stale/quantiles_overflow_ranks"] == 0
    monitor.add(torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 2), torch.ones(1, 2))
    actual = monitor.metrics()
    assert actual["stale/selected_tokens"] == 7
    assert actual["stale/finite_tokens"] == 6
    assert actual["stale/finite_fraction"] == 6 / 7
    assert actual["stale/statistics_valid"] == 0
    assert actual["stale/quantiles_valid"] == 0
    assert actual["stale/abs_log_ratio_p50"] == 0
