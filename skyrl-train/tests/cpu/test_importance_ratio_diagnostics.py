# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-rank and cross-mini-batch reduction of the training status dict.

At E6 geometry the PPO ratio is an invariant, not a statistic: policy_mini_batch_size equals
train_batch_size with one optimizer update per step, so the training forward and the old-logprob
forward run on identical weights and every token's ratio must be exactly one. PR #488 reports the
same at its geometry. That makes log_ratio_abs_max the cheapest correctness gate available -- and it
is only a gate if BOTH reductions that carry it, across ranks and across a step's mini-batches,
preserve a max.
"""

from unittest.mock import patch

import pytest
import torch

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.utils.importance_ratio_diagnostics import (
    MAX_REDUCED_METRIC_KEYS,
    STATUS_REDUCTION_OPS,
    SUM_REDUCED_METRIC_KEYS,
)
from skyrl_train.utils.metrics import policy_training_metrics


class _StubbedStrategy(DistributedStrategy):
    """The real all_reduce against a stubbed collective.

    Substituting a fake reducer would test the fake: ``op="maximum"`` would pass every assertion here
    and fail at runtime on Strategy.all_reduce's own assert. This keeps the production branch that
    skips ``data /= world_size`` for max, and the op it hands to the collective.
    """

    # The abstract surface is training lifecycle; none of it is reachable from all_reduce_status.
    backward = load_checkpoint = optimizer_step = save_checkpoint = save_hf_model = setup_distributed = (
        lambda self, *args, **kwargs: None
    )

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size


def _reduce_one(key, this_rank, peers, ops_seen):
    """Reduce a single key as if `peers` were the other ranks' contributions.

    One key per call because a mean and a sum both arrive at the collective as ReduceOp.SUM -- the
    difference is the local divide that has already happened -- so a shared stub cannot tell them
    apart.
    """
    strategy = _StubbedStrategy(1 + len(peers))

    def _capture(tensor, op=None):
        ops_seen.append(op)
        mine = float(tensor.item())
        if op is torch.distributed.ReduceOp.MAX:
            tensor.fill_(max([mine, *peers]))
        else:
            # Every peer applies the same local scaling this rank did before contributing.
            scale = mine / this_rank if this_rank else 1.0
            tensor.fill_(mine + sum(peer * scale for peer in peers))
        return None

    # all_reduce moves CPU scalars to the current CUDA device before the collective; on a CPU-only
    # runner that is stubbed so the real reduction logic still runs.
    with (
        patch("skyrl_train.distributed.strategy.dist.all_reduce", side_effect=_capture),
        patch("skyrl_train.distributed.strategy.torch.cuda.current_device", return_value="cpu"),
    ):
        return strategy.all_reduce_status({key: this_rank})[key]


def test_a_max_valued_metric_reaches_the_real_reduce_op_max():
    """🚨 A max folded into a mean cannot see a divergence confined to a few ranks.

    Each rank's log_ratio_abs_max is a true local max, so meaning them across 80 ranks turns one rank
    at 19.0 with 79 clean into 0.2375 -- which reads as zero on any dashboard. This drives the real
    Strategy.all_reduce so the op that reaches the collective is the one asserted.
    """
    ops: list = []
    assert _reduce_one("log_ratio_abs_max", 19.0, [0.0, 0.0, 0.0], ops) == 19.0, (
        "the divergent rank must survive; a mean of 19/0/0/0 publishes 4.75"
    )
    assert ops == [torch.distributed.ReduceOp.MAX], "the max key must reach ReduceOp.MAX"

    ops.clear()
    assert _reduce_one("policy_loss", 1.0, [3.0, 3.0, 3.0], ops) == pytest.approx(2.5), (
        "everything else keeps the mean it has always had"
    )
    assert ops == [torch.distributed.ReduceOp.SUM], "a mean is a local divide plus a SUM"


def test_a_token_count_is_summed_across_ranks_not_averaged():
    """The gt_*pct keys are token COUNTS. A mean divides the global count by world size, so 1000
    offending tokens on one of 80 ranks publishes 12.5 -- nonzero, so an alert still fires, but every
    magnitude 80x low and the series is not a token count."""
    assert _reduce_one("n_tokens_dp_gt_1pct", 1000.0, [0.0, 0.0, 0.0], []) == 1000.0, (
        "a mean would publish 250.0 for a count of 1000"
    )


def test_the_two_reduction_axes_share_one_op_map():
    """Reducing correctly across ranks and then averaging back down over mini-batches is the same
    category error one level lower: at n mini-batches a max becomes mean-of-n-maxima and a "did any
    rank fail" flag becomes 1/n. E6 has one mini-batch per step and would not notice."""
    status = policy_training_metrics(
        {
            "log_ratio_abs_max": [19.0, 0.0, 0.0, 0.0],
            "log_ratio_diagnostics_failed": [1.0, 0.0, 0.0, 0.0],
            "n_tokens_dp_gt_1pct": [10.0, 20.0],
            "policy_loss": [1.0, 3.0],
        },
        policy_update_steps=1.0,
    )
    assert status["log_ratio_abs_max"] == 19.0, "a mean over mini-batches would publish 4.75"
    assert status["log_ratio_diagnostics_failed"] == 1.0, "any mini-batch failing is a failure"
    assert status["n_tokens_dp_gt_1pct"] == 30.0
    assert status["policy_loss"] == 2.0


def test_every_specially_reduced_key_is_declared_once():
    """One map, so the two axes cannot drift apart."""
    assert set(STATUS_REDUCTION_OPS) == set(MAX_REDUCED_METRIC_KEYS) | set(SUM_REDUCED_METRIC_KEYS)
    assert all(STATUS_REDUCTION_OPS[key] == "max" for key in MAX_REDUCED_METRIC_KEYS)
    assert all(STATUS_REDUCTION_OPS[key] == "sum" for key in SUM_REDUCED_METRIC_KEYS)
    # p99 is deliberately absent: a max of per-rank p99 approximations is not a quantile either.
    assert "log_ratio_abs_p99" not in STATUS_REDUCTION_OPS
