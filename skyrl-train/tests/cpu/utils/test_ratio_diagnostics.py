import math
import json
from pathlib import Path

import numpy
import pytest
import torch

from skyrl_train.utils.importance_ratio_diagnostics import (
    LogRatioMonitor,
    absolute_quantiles,
    mismatch_ratio_metrics,
    ratio_statistics,
    gather_ratio_tensor,
)


def test_ratio_statistics_match_hand_values_and_clamp_only_exponentials():
    values = [0, math.log(2), -math.log(2), math.log(4), math.log(1e-6), 0]
    result = ratio_statistics(torch.tensor(values, dtype=torch.float64))
    assert result["frac_outside_0_5_2"] == pytest.approx(2 / 6)
    assert result["frac_below_1e_5"] == pytest.approx(1 / 6)
    assert result["kl_k1"] == pytest.approx(-sum(values) / 6, abs=1e-9)
    assert result["kl_k3"] == pytest.approx(sum(math.exp(x) - x - 1 for x in values) / 6, abs=1e-9)
    assert result["chi2"] == pytest.approx(sum(math.exp(2 * x) for x in values) / 6 - 1, abs=1e-9)
    extreme = ratio_statistics(torch.tensor([1000.0], dtype=torch.float64))
    assert extreme["log_ratio_abs_mean"] == 1000
    assert extreme["log_ratio_mean_squared"] == 1e6
    assert extreme["ess_fraction"] == 1
    assert all(math.isfinite(value) for value in extreme.values())


def test_mismatch_metrics_bucket_by_staleness_and_absolute_position():
    lengths = torch.tensor([300, 600, 100, 700])
    mask = torch.arange(700).unsqueeze(0) < lengths.unsqueeze(1)
    delta = torch.tensor([0.1, 0.2, 0.3, 0.9], dtype=torch.float64).unsqueeze(1).expand(4, 700)
    rollout = torch.zeros_like(delta)
    rollout[~mask] = math.nan
    result = mismatch_ratio_metrics(delta, rollout, mask, torch.tensor([0, 0, 3, 9]))
    assert result["policy/mismatch/staleness0/log_ratio_abs_mean"] == pytest.approx((300 * 0.1 + 600 * 0.2) / 900)
    assert result["policy/mismatch/staleness8+/log_ratio_abs_mean"] == pytest.approx(0.9)
    assert result["policy/mismatch/staleness0/pos_last256/selected_tokens"] == 512
    assert result["policy/mismatch/staleness3/pos_first256/selected_tokens"] == 100
    assert result["policy/mismatch/staleness3/pos_last256/selected_tokens"] == 100
    assert result["policy/mismatch/staleness1/selected_tokens"] == 0
    assert "policy/mismatch/staleness1/ess_fraction" not in result
    # Put a single mismatch exactly at the 600-token row's last-window boundary.
    changed = torch.zeros_like(delta)
    changed[1, 343:345] = torch.tensor([10.0, 20.0])
    boundary = mismatch_ratio_metrics(changed, rollout, mask, torch.tensor([0, 0, 3, 9]))
    assert boundary["policy/mismatch/staleness0/pos_last256/log_ratio_abs_mean"] == pytest.approx(20 / 512)


# Every policy/mismatch key the async RL Grafana dashboard reads
# (marin infra/grafana/src/async_rl_observability.py).
DASHBOARD_MISMATCH_KEYS = (
    # Panel 30, pre-update model log-ratio drift.
    "pooled/log_ratio_mean",
    "pooled/log_ratio_abs_mean",
    "pooled/log_ratio_abs_p95",
    "pooled/log_ratio_abs_p99",
    "pooled/log_ratio_abs_max",
    # Panel 31, PPO-window pressure.
    "pooled/lower_clip_pressure",
    "pooled/upper_clip_pressure",
    # Panels 32, 47 and 48: coverage, ESS and the uniform-staleness tables.
    "pooled/finite_fraction",
    "pooled/missing_behavior",
    "pooled/ess_fraction",
    # Panels 44, 47 and 48, mean squared log-ratio.
    "pooled/log_ratio_mean_squared",
    # Panel 54, staleness-zero mismatch.
    "staleness0/log_ratio_abs_mean",
    "staleness0/log_ratio_abs_p99",
    "staleness0/log_ratio_abs_p999",
    "staleness0/frac_outside_0_5_2",
    "staleness0/ess_fraction",
    "staleness0/kl_k3",
    "staleness0/chi2",
    # Panel 55, mismatch by staleness bucket.
    *(f"staleness{bucket}/log_ratio_abs_mean" for bucket in ("0", "1", "2", "3", "4-7", "8+")),
    # Panel 57, position dependence.
    "pooled/pos_first256/log_ratio_abs_mean",
    "pooled/pos_last256/log_ratio_abs_mean",
    "pooled/pos_middle/log_ratio_abs_mean",
)


@pytest.mark.parametrize("staleness", [[0, 1, 2, 3, 5, 9], [0, 0, 0, 0, 0, 0]])
def test_mismatch_metrics_emit_every_key_the_dashboard_reads(staleness):
    mask = torch.ones(6, 800)
    learner = torch.randn(6, 800, dtype=torch.float64)
    result = mismatch_ratio_metrics(learner, torch.zeros_like(learner), mask, torch.tensor(staleness))
    read = {f"policy/mismatch/{key}" for key in DASHBOARD_MISMATCH_KEYS if staleness[-1] or "staleness0" in key}
    assert read - result.keys() == set()
    assert all(math.isfinite(result[key]) for key in read)
    staleness0 = ratio_statistics(learner[torch.tensor(staleness) == 0].reshape(-1))
    assert {key: result[f"policy/mismatch/staleness0/{key}"] for key in staleness0} == pytest.approx(staleness0)


def test_worker_accumulator_matches_pooled_ess_and_tail_under_unequal_microbatches():
    values = torch.linspace(-10, 10, 3000, dtype=torch.float64)
    monitor = LogRatioMonitor(torch.device("cpu"))
    for shard in (values[:100], values[100:1200], values[1200:]):
        shard = shard.unsqueeze(0)
        monitor.add(shard, torch.zeros_like(shard), torch.ones_like(shard))
    actual, expected = monitor.metrics(), ratio_statistics(values)
    assert actual["log_ratio_ess_fraction"] == pytest.approx(expected["ess_fraction"], abs=1e-10)
    assert actual["log_ratio_abs_p999"] == pytest.approx(expected["log_ratio_abs_p999"], abs=1e-5)
    assert actual["log_ratio_mean"] == pytest.approx(expected["log_ratio_mean"], abs=1e-10)
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
    assert actual["log_ratio_chi2"] == pytest.approx(math.exp(2 * 20) - 1)
    assert actual["log_ratio_statistics_valid"] == 1


def _distributed_ratio_worker(rank, directory):
    torch.distributed.init_process_group("gloo", init_method=f"file://{directory}/group", rank=rank, world_size=2)
    try:
        values = torch.arange(1, 101, dtype=torch.float64) / 10 if rank == 0 else torch.zeros(19_900)
        monitor = LogRatioMonitor(torch.device("cpu"))
        values = values.unsqueeze(0)
        monitor.add(values, torch.zeros_like(values), torch.ones_like(values))
        pooled = monitor.metrics(gather_fn=gather_ratio_tensor)
        if rank == 1:
            monitor._failed = True
        failed = monitor.metrics(gather_fn=gather_ratio_tensor)
        Path(directory, f"rank{rank}.json").write_text(json.dumps({"pooled": pooled, "failed": failed}))
    finally:
        torch.distributed.destroy_process_group()


def test_two_actual_gloo_ranks_emit_identical_token_pooled_statistics(tmp_path):
    torch.multiprocessing.spawn(_distributed_ratio_worker, args=(str(tmp_path),), nprocs=2, join=True)
    left, right = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert left == right
    expected = ratio_statistics(torch.cat([torch.arange(1, 101, dtype=torch.float64) / 10, torch.zeros(19_900)]))
    assert left["pooled"]["log_ratio_mean"] == pytest.approx(expected["log_ratio_mean"], abs=1e-10)
    assert left["pooled"]["log_ratio_ess_fraction"] == pytest.approx(expected["ess_fraction"], rel=1e-10)
    assert left["pooled"]["log_ratio_abs_p999"] == pytest.approx(expected["log_ratio_abs_p999"], abs=1e-5)
    assert left["pooled"]["log_ratio_p999_valid"] == 1
    assert left["failed"]["log_ratio_p999_valid"] == 0
    assert left["failed"]["log_ratio_diagnostics_failed"] == 1


def test_a_nonfinite_token_invalidates_the_worker_statistics():
    values = torch.tensor([[-2.0, -0.25, 0.0, 0.5, 3.0]], dtype=torch.float64)
    monitor = LogRatioMonitor(torch.device("cpu"))
    monitor.add(values, torch.zeros_like(values), torch.ones_like(values))
    assert monitor.metrics()["log_ratio_statistics_valid"] == 1
    monitor.add(torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 2), torch.ones(1, 2))
    assert monitor.metrics()["log_ratio_statistics_valid"] == 0


def test_quantiles_stay_exact_above_the_size_torch_refuses():
    """numpy is the independent reference: torch.quantile refuses this population outright."""
    values = torch.rand(2**24 + 1, dtype=torch.float32)
    probabilities = (0.5, 0.95, 0.99)
    with pytest.raises(RuntimeError):
        torch.quantile(values, values.new_tensor(probabilities))
    expected = numpy.quantile(values.numpy(), probabilities, method="linear")
    assert absolute_quantiles(values, probabilities) == pytest.approx(expected, abs=1e-6)
    small = torch.rand(1000, dtype=torch.float64)
    expected = torch.quantile(small, small.new_tensor(probabilities)).tolist()
    assert absolute_quantiles(small, probabilities) == pytest.approx(expected, rel=1e-12)
    ties = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    tied = torch.quantile(ties, ties.new_tensor(probabilities)).tolist()
    assert absolute_quantiles(ties, probabilities) == pytest.approx(tied, rel=1e-12)
