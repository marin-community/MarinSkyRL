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
        elif op is torch.distributed.ReduceOp.MIN:
            tensor.fill_(min([mine, *peers]))
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


def test_no_worker_reduces_a_status_dict_with_the_plain_mean():
    """STATUS_REDUCTION_OPS is keyed by metric NAME, not by worker.

    A `all_reduce(status)` left anywhere means the first key that worker ever shares with the policy
    is silently meaned -- and the branch added AST walks for span names and for manual context
    managers, so leaving this one unguarded is inconsistent with its own convention. The critic path
    carried it: latent, because no shared key exists today, which is precisely why it would go
    unnoticed when one appears.
    """
    import ast
    import inspect

    import skyrl_train.workers.worker as worker_module

    offenders: list[str] = []
    for node in ast.walk(ast.parse(inspect.getsource(worker_module))):
        if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "all_reduce":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id == "status":
                offenders.append(f"line {node.lineno}")
    assert not offenders, (
        f"all_reduce(status) at {offenders}: a status dict must go through all_reduce_status, or "
        "its specially-reduced keys silently take the default mean"
    )


def test_a_whole_status_dict_gives_every_key_its_own_op():
    """The shape production actually uses: one dict, several keys, three different ops.

    🚨 Every other test here reduces ONE key per call, so `all_reduce_status`'s grouping loop was
    never exercised -- reducing every group with the first group's op left the entire
    suite green. That would silently mean `log_ratio_abs_max` whenever an ordinary metric sorted
    first, publishing 0.2375 for one rank at 19.0 among eighty, which is the exact number this
    branch exists to stop being invisible.
    """
    ops_by_value: dict[float, object] = {}

    class _Recording(_StubbedStrategy):
        def all_reduce(self, data, op="mean"):
            # Record the op each KEY was reduced with, not merely the ops seen.
            for name in data:
                ops_by_value[name] = op
            return {name: value for name, value in data.items()}

    status = {
        "policy_loss": 1.0,  # mean, the default
        "log_ratio_abs_max": 19.0,  # max
        "log_ratio_diagnostics_failed": 1.0,  # max
        "optimizer_step_succeeded": 0.0,  # min
    }
    _Recording(4).all_reduce_status(status)

    assert ops_by_value == {
        "policy_loss": "mean",
        "log_ratio_abs_max": "max",
        "log_ratio_diagnostics_failed": "max",
        "optimizer_step_succeeded": "min",
    }


def test_a_min_valued_metric_reaches_the_real_reduce_op_min():
    """Drives the real Strategy.all_reduce, so the op that reaches the collective is what is asserted.

    The map-inspecting test next door cannot see this: dispatching SUM specifically for op == "min"
    while leaving REDUCE_OPS["min"] == MIN passes it. optimizer_step_succeeded is a binary
    did-every-rank-succeed flag, so one failing rank in four must publish 0 -- a mean publishes 0.75,
    and a SUM publishes 3, which at 80 healthy ranks is 80.0 next to a threshold of 1.
    """
    ops: list = []
    assert _reduce_one("optimizer_step_succeeded", 1.0, [1.0, 0.0, 1.0], ops) == 0.0, (
        "one failing rank must survive the reduction; a mean publishes 0.75 and a sum publishes 3"
    )
    assert ops == [torch.distributed.ReduceOp.MIN], "the min key must reach ReduceOp.MIN"


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


def test_a_replicated_token_count_is_not_multiplied_by_the_replication_factor():
    """🚨 A sum over WORLD is not a global count when ranks replicate.

    Strategy.all_reduce reduces over WORLD. Under sequence, context, expert or Megatron
    tensor/pipeline parallelism the replicas hold the SAME tokens, so summing multiplies the count
    by the replication factor -- 16x at CP2xEP8. A mean reads low by the world size instead, which
    is a documented per-rank average rather than a wrong global total. Neither op is right; a
    correct count needs a data-parallel-group reduction this primitive cannot express.
    """
    assert "n_tokens_dp_gt_1pct" not in STATUS_REDUCTION_OPS, (
        "summing this over WORLD overcounts every replicated topology"
    )
    # Two ranks replicating the same 100 tokens: the mean is 100, a sum would be 200.
    assert _reduce_one("n_tokens_dp_gt_1pct", 100.0, [100.0], []) == pytest.approx(100.0)


def test_the_two_reduction_axes_share_one_op_map():
    """Reducing correctly across ranks and then averaging back down over mini-batches is the same
    category error one level lower: at n mini-batches a max becomes mean-of-n-maxima and a "did any
    rank fail" flag becomes 1/n. E6 has one mini-batch per step and would not notice."""
    status = policy_training_metrics(
        {
            "log_ratio_abs_max": [19.0, 0.0, 0.0, 0.0],
            "log_ratio_diagnostics_failed": [1.0, 0.0, 0.0, 0.0],
            "policy_loss": [1.0, 3.0],
        },
        policy_update_steps=1.0,
    )
    assert status["log_ratio_abs_max"] == 19.0, "a mean over mini-batches would publish 4.75"
    assert status["log_ratio_diagnostics_failed"] == 1.0, "any mini-batch failing is a failure"
    assert status["policy_loss"] == 2.0


def test_the_rank_axis_maps_min_to_the_torch_min_op():
    """The other axis of the same reduction, pinned to the real torch op rather than a string.

    all_reduce moves its tensor to the current CUDA device before reducing, so the dispatch itself is
    unreachable on a CPU runner -- swapping ReduceOp.MIN for SUM there would otherwise change no
    test while turning "did every rank succeed" into "how many did", which reads as 80 where 1 was
    the healthy value.
    """
    import torch.distributed as dist

    from skyrl_train.distributed.strategy import REDUCE_OPS

    assert REDUCE_OPS["min"] is dist.ReduceOp.MIN
    assert REDUCE_OPS["max"] is dist.ReduceOp.MAX
    # mean and sum both dispatch to SUM: mean's division by world_size happens locally, before the
    # collective. They are in the map because the lookup is now a direct index with NO default. An
    # earlier version fell back to SUM for anything missing, under a comment claiming that fallback
    # had been removed -- so a min quietly becoming a SUM would publish 80.0 for a healthy step at
    # 80 ranks, and nothing would have failed.
    assert REDUCE_OPS["sum"] is dist.ReduceOp.SUM
    assert REDUCE_OPS["mean"] is dist.ReduceOp.SUM

    # The map must COVER every op all_reduce accepts, or that direct index raises KeyError at
    # collective time -- on a real run, at 80 ranks, in the step epilogue.
    import inspect
    import re as _re

    from skyrl_train.distributed.strategy import DistributedStrategy

    guard = _re.search(r"assert op in \(([^)]*)\)", inspect.getsource(DistributedStrategy.all_reduce))
    assert guard, "all_reduce no longer declares the ops it accepts; this check cannot see them"
    for op in _re.findall(r'"(\w+)"', guard.group(1)):
        assert op in REDUCE_OPS, f"all_reduce accepts {op!r} but REDUCE_OPS has no collective for it"

    # Every op the status map declares must be one the rank axis can perform.
    for op in set(STATUS_REDUCTION_OPS.values()):
        assert op in REDUCE_OPS, f"{op!r} has no rank-axis implementation"


def test_the_mini_batch_axis_takes_a_min_for_min_reduced_keys():
    """Behavioural, not declarative. The declaration test checks the MAP; this checks the ARITHMETIC.

    optimizer_step_succeeded is a binary did-every-rank-succeed flag. Over a step's mini-batches, one
    failed update among four must publish 0 -- a mean publishes 0.75 and a sum publishes 3, and both
    read as "fine" next to a threshold of 1.
    """
    status = policy_training_metrics(
        {"optimizer_step_succeeded": [1.0, 1.0, 0.0, 1.0], "policy_loss": [1.0, 3.0]},
        policy_update_steps=4.0,
    )
    assert status["optimizer_step_succeeded"] == 0.0, "one failed update in the step is a failed step"
    assert status["policy_loss"] == 2.0, "ordinary metrics still average"


def test_the_mini_batch_axis_refuses_an_op_it_cannot_perform():
    """The earlier form was `max(values) if op == "max" else sum(values)`.

    Any op it did not know became a SUM, silently. Adding a third op to the map would then have
    published a sum of flags and looked like a number.
    """
    from skyrl_train.utils import metrics as metrics_module

    original = dict(metrics_module.STATUS_REDUCTION_OPS)
    metrics_module.STATUS_REDUCTION_OPS["made_up_key"] = "median"
    try:
        with pytest.raises(ValueError, match="unknown reduction op"):
            policy_training_metrics({"made_up_key": [1.0, 2.0]}, policy_update_steps=1.0)
    finally:
        metrics_module.STATUS_REDUCTION_OPS.clear()
        metrics_module.STATUS_REDUCTION_OPS.update(original)


def test_the_specially_reduced_path_rejects_a_non_numeric_value():
    """mean_metrics rejects them; the max/min path dropped that check.

    A 0-d tensor would otherwise be published where a float is expected, on the keys designated
    gate-grade -- and it would compare and format without complaining.
    """
    with pytest.raises(TypeError, match="non-numeric"):
        policy_training_metrics({"log_ratio_abs_max": [1.0, object()]}, policy_update_steps=1.0)


def test_every_specially_reduced_key_is_declared_once():
    """One map, so the two axes cannot drift apart."""
    from skyrl_train.utils.importance_ratio_diagnostics import MIN_REDUCED_METRIC_KEYS

    assert set(STATUS_REDUCTION_OPS) == set(MAX_REDUCED_METRIC_KEYS) | set(MIN_REDUCED_METRIC_KEYS)
    assert all(STATUS_REDUCTION_OPS[key] == "max" for key in MAX_REDUCED_METRIC_KEYS)
    assert all(STATUS_REDUCTION_OPS[key] == "min" for key in MIN_REDUCED_METRIC_KEYS)
    # No key may be declared under two ops; the whole point of one map is that the rank axis and the
    # mini-batch axis cannot disagree about a key's meaning.
    assert not set(MAX_REDUCED_METRIC_KEYS) & set(MIN_REDUCED_METRIC_KEYS)
    # Every op in the map must be one the reducers actually implement, on BOTH axes. An op only one
    # axis knows is how a key gets reduced two different ways.
    assert set(STATUS_REDUCTION_OPS.values()) <= {"max", "min", "sum"}
    # Deliberately absent, because neither available op is right for them.
    assert "log_ratio_abs_p99" not in STATUS_REDUCTION_OPS
    assert "n_tokens_dp_gt_1pct" not in STATUS_REDUCTION_OPS
