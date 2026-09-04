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

    ⚠️ **This walk is the ONLY guard available for the Megatron site, and that is a constraint rather
    than a choice.** `megatron_worker.py` imports `megatron.bridge` and `megatron.core` at module
    scope, and megatron is not installed in the CPU environment -- so the behavioural harness that
    covers `PolicyWorkerBase.ppo_train` (`_run_ppo_train`, in test_policy_train_spans.py) cannot be
    built for `MegatronPolicyWorkerBase.ppo_train` here. The walk therefore has to be as strong as a
    static check can be: it resolves ALIASES, because a bare identifier match let
    `metrics = status; all_reduce(metrics)` through in two separate review rounds.

    A `all_reduce(status)` left anywhere means the first key that worker ever shares with the policy
    is silently meaned -- and the branch added AST walks for span names and for manual context
    managers, so leaving this one unguarded is inconsistent with its own convention. The critic path
    carried it: latent, because no shared key exists today, which is precisely why it would go
    unnoticed when one appears.
    """
    import ast
    import pathlib

    import skyrl_train.workers.worker as worker_module

    def _status_aliases(tree: ast.AST) -> set[str]:
        """Every identifier bound to the status dict, so a rename cannot walk past this.

        ⚠️ `metrics = status; all_reduce(metrics)` preserves the defect exactly and changes only the
        spelling, and a walk keyed on the identifier `status` accepted it -- verified surviving the
        whole suite twice, in two separate review rounds.
        """
        names = {"status"}
        for _ in range(3):  # a chain `a = status; b = a` needs one pass per link
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                value = node.value
                bound = (isinstance(value, ast.Name) and value.id in names) or (
                    isinstance(value, ast.Attribute) and value.attr in names
                )
                if not bound:
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        # ⚠️ Attribute targets too. `self._pending = status;
                        # all_reduce(self._pending)` is the same defect held in a field, and a
                        # Name-only walk let it through on the critic path.
                        names.add(target.attr)
        return names

    def _mentions_status(node: ast.AST, names: set[str]) -> bool:
        # Any subexpression naming the dict counts: `all_reduce(dict(status))` and
        # `all_reduce(data=status)` are the same defect as the bare name.
        return any(
            (isinstance(inner, ast.Name) and inner.id in names)
            or (isinstance(inner, ast.Attribute) and inner.attr in names)
            for inner in ast.walk(node)
        )

    # ⚠️ The WHOLE workers package, not `inspect.getsource(worker_module)`. This branch changed the
    # same call in workers/megatron/megatron_worker.py, and a walk scoped to one file left that one
    # revertible in silence. Scope the guard to every file the defect can live in, not to the file
    # where it was first found.
    workers = pathlib.Path(worker_module.__file__).parent
    sources = sorted(workers.rglob("*.py"))
    assert len(sources) > 1, "the walk must cover the whole workers package, not a single module"

    offenders: list[str] = []
    for path in sources:
        tree = ast.parse(path.read_text())
        names = _status_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or getattr(node.func, "attr", None) != "all_reduce":
                continue
            if any(_mentions_status(arg, names) for arg in (*node.args, *(kw.value for kw in node.keywords))):
                offenders.append(f"{path.relative_to(workers)}:{node.lineno}")
    assert not offenders, (
        f"all_reduce(status) at {offenders}: a status dict must go through all_reduce_status, or "
        "its specially-reduced keys silently take the default mean"
    )


def test_a_whole_status_dict_gives_every_key_its_own_op():
    """The shape production actually uses: one dict, several keys, three different ops.

    🚨 Every other test here reduces ONE key per call, so `all_reduce_status`'s grouping loop was
    never exercised -- reducing every group with the first group's op left the whole suite green. That would silently mean `log_ratio_abs_max` whenever an ordinary metric sorted
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

    original = dict(metrics_module.MINI_BATCH_REDUCTION_OPS)
    metrics_module.MINI_BATCH_REDUCTION_OPS["made_up_key"] = "median"
    try:
        with pytest.raises(ValueError, match="unknown reduction op"):
            policy_training_metrics({"made_up_key": [1.0, 2.0]}, policy_update_steps=1.0)
    finally:
        metrics_module.MINI_BATCH_REDUCTION_OPS.clear()
        metrics_module.MINI_BATCH_REDUCTION_OPS.update(original)


def test_a_ranks_token_counts_are_SUMMED_across_its_optimizer_windows():
    """🚨 The two reduction axes disagree here, and one map could not express it.

    Across RANKS these stay a mean: replicas under sequence/context/expert/tensor parallelism hold
    the SAME tokens, so a WORLD sum multiplies the count by the replication factor (16x at CP2xEP8).
    Across MINI-BATCHES the opposite holds -- one rank's optimizer windows hold DIFFERENT tokens --
    so a mean is a category error. Windows of 3 and 7 offending tokens published 5.0 where the
    rank's step total is 10; on a 32-window step that understates the count by roughly 32x.
    """
    from skyrl_train.utils.importance_ratio_diagnostics import (
        MINI_BATCH_REDUCTION_OPS,
        STATUS_REDUCTION_OPS,
        TOKEN_COUNT_METRIC_KEYS,
    )

    status = policy_training_metrics(
        {key: [3.0, 7.0] for key in TOKEN_COUNT_METRIC_KEYS},
        policy_update_steps=2.0,
    )
    for key in TOKEN_COUNT_METRIC_KEYS:
        assert status[key] == 10.0, f"{key} published {status[key]}, the mean, not the step total"

    # And the rank axis is deliberately NOT changed: a sum there would multiply replicated tokens.
    for key in TOKEN_COUNT_METRIC_KEYS:
        assert key not in STATUS_REDUCTION_OPS, (
            f"{key} must stay mean-reduced across ranks; a WORLD sum multiplies by the replication "
            "factor, which is why the two axes need separate maps"
        )
        assert MINI_BATCH_REDUCTION_OPS[key] == "sum"

    # Everything else still agrees across the two axes, so drift takes an explicit decision.
    shared = {k: v for k, v in MINI_BATCH_REDUCTION_OPS.items() if k not in TOKEN_COUNT_METRIC_KEYS}
    assert shared == STATUS_REDUCTION_OPS


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


def _training_input_batch(batch: int, seq: int, actions: int):
    """The tensors TrainingBatchIterator reads to build one Experience."""
    import torch

    from skyrl_train.training_batch import TrainingInputBatch

    data = TrainingInputBatch(
        {
            "sequences": torch.zeros(batch, seq, dtype=torch.long),
            "action_log_probs": torch.zeros(batch, actions),
            "base_action_log_probs": torch.zeros(batch, actions),
            "values": torch.zeros(batch, actions),
            "returns": torch.zeros(batch, actions),
            "advantages": torch.zeros(batch, actions),
            "attention_mask": torch.ones(batch, seq, dtype=torch.long),
            "loss_mask": torch.ones(batch, actions),
            "response_mask": torch.ones(batch, actions),
        }
    )
    data.metadata = {"global_step": 0, "response_length": actions}
    return data


def _megatron_worker_module():
    """Import the Megatron worker on CPU, extending the house stub rather than replacing it.

    ⚠️ An earlier docstring in this file asserted a behavioural test here was IMPOSSIBLE, because
    `megatron_worker` imports `megatron.bridge` and `megatron.core` at module scope and megatron is
    not installed on CPU. **That was wrong**, and two reviewers said so. The imports only need to
    resolve; nothing in `ppo_train` calls into megatron, because the pipeline scheduler is reached
    through `self.model`. Asserting an impossibility is how a guard stays lexical forever.

    ⚠️ And it must EXTEND `tests.cpu.util.stub_megatron_modules`, not install a rival mechanism.
    A first version fabricated its own `megatron.*` via a meta-path finder, which worked alone and
    failed whenever `test_tis_diagnostics_backends` had already installed the house stub -- the
    module object was then present but not a package, so `megatron.bridge` could not resolve. An
    order-dependent test is worse than no test: it passes in isolation and blames the wrong change.
    """
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    from tests.cpu.util import stub_megatron_modules

    stub_megatron_modules()

    class _Meta(type):
        # Class-level attribute access, which instance __getattr__ does not cover. Megatron's
        # bridges are used as decorators off the CLASS (`MegatronModelBridge.register_bridge`).
        def __getattr__(cls, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Perm

    class _Perm(metaclass=_Meta):
        """Stands in for a megatron symbol: callable, subclassable, decorator-friendly.

        A real class rather than a MagicMock, because `@dataclass` on a subclass needs a genuine
        `__mro__` and a mock has none.
        """

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return _Perm

        def __getattr__(self, name):
            return _Perm

    class _PermModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            setattr(self, name, _Perm)
            return _Perm

    # The house stub installs plain ModuleType objects with no __path__, so a dotted import beneath
    # them fails with "is not a package". Make every already-installed megatron module a package and
    # give it the permissive attribute behaviour megatron_worker needs.
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in ("megatron", "transformer_engine") and isinstance(module, types.ModuleType):
            if not hasattr(module, "__path__"):
                module.__path__ = []
            if type(module) is types.ModuleType:
                module.__class__ = _PermModule

    # Everything else beneath those roots is fabricated on demand. Scoped to the two roots, defers
    # to whatever the house stub already installed, and removed again below -- a finder left on
    # sys.meta_path would stub megatron for every later test in the session.
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in sys.modules:
                return None
            if fullname.split(".")[0] not in ("megatron", "transformer_engine"):
                return None
            return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            module = _PermModule(spec.name)
            module.__path__ = []
            return module

        def exec_module(self, module):
            pass

    # Link every parent to its child BEFORE anything reads an attribute. _PermModule caches what
    # __getattr__ returns, so one early `megatron.core` lookup would pin the placeholder class in
    # place of the real stub module and `import megatron.core.parallel_state` would then fail.
    def _link() -> None:
        for name in sorted(sys.modules):
            if name.split(".")[0] not in ("megatron", "transformer_engine") or "." not in name:
                continue
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, sys.modules[name])

    _link()

    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        import skyrl_train.workers.megatron.megatron_worker as module

        _link()
    finally:
        sys.meta_path.remove(finder)

    return module


def test_the_MEGATRON_worker_keeps_its_per_key_ops_through_its_own_ppo_train(monkeypatch):
    """🚨 The Megatron reduction site, driven -- not read.

    Its `all_reduce_status` call was guarded only by a source walk, and two review rounds each
    walked past it: first with a bare alias, then with `status.copy()`. On eighty ranks that path
    turns one rank's `log_ratio_abs_max` of 19.0 into 0.2375 and a failed optimizer step into 0.9875,
    which is the exact failure the reduction map exists to prevent.
    """
    import torch
    from omegaconf import OmegaConf

    module = _megatron_worker_module()
    worker = object.__new__(module.MegatronPolicyWorkerBase)

    peer = {"log_ratio_abs_max": 19.0, "optimizer_step_succeeded": 0.0}

    class _Strategy:
        def is_rank_0(self):
            return True

        def all_reduce_status(self, status):
            from skyrl_train.utils.importance_ratio_diagnostics import STATUS_REDUCTION_OPS

            out = {}
            for name, value in status.items():
                other = peer.get(name, value)
                op = STATUS_REDUCTION_OPS.get(name)
                out[name] = (
                    max(value, other) if op == "max" else min(value, other) if op == "min" else (value + other) / 2
                )
            return out

        def all_reduce(self, data, op="mean"):
            return {name: (value + peer.get(name, value)) / 2 for name, value in data.items()}

        def optimizer_step(self, *a, **k):
            return 0.1

    class _Model:
        def train(self):
            pass

        def forward_backward_mini_batch(self, micro_batches, **kwargs):
            return [
                {
                    "policy_loss": 1.0,
                    "policy_entropy": 0.5,
                    # popped when use_kl_loss is false; present because the real metrics carry it
                    "policy_kl": 0.0,
                    "log_ratio_abs_max": 0.5,
                    "optimizer_step_succeeded": 1.0,
                }
                for _ in micro_batches
            ]

    worker.cfg = OmegaConf.create(
        {
            "trainer": {
                "micro_train_batch_size_per_gpu": 1,
                "update_epochs_per_batch": 1,
                "algorithm": {"use_kl_loss": False},
                "policy": {"megatron_config": {"check_train_eval_parity": False}},
            },
            "generator": {"sampling_params": {"temperature": 1.0}},
        }
    )
    worker.strategy = _Strategy()
    worker.model = _Model()
    worker.actor_module = []

    class _Optimizer:
        param_groups = [{"lr": 1e-6}]

        def zero_grad(self, *a, **k):
            pass

    worker.optimizer = _Optimizer()
    worker.scheduler = None
    worker.profiler = None
    worker.empty_cuda_cache = False
    worker.policy_mini_batch_size_per_gpu = 1
    worker._rank = 0

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    # ppo_train closes with a WORLD barrier; there is no process group in a CPU test.
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)

    batch, seq, actions = 1, 4, 2
    data = _training_input_batch(batch, seq, actions)
    try:
        output = worker.ppo_train(data)
    except Exception as exc:  # noqa: BLE001 - reported, never skipped
        pytest.fail(f"the Megatron worker needs more scaffolding than this fake provides: {exc!r}")

    status = output.metadata["train_status"]
    assert status["log_ratio_abs_max"] == 19.0, (
        f"published {status['log_ratio_abs_max']}: the peer rank's divergence was averaged away, "
        "which is what the plain reducer does and what the status reducer exists to prevent"
    )
    assert status["optimizer_step_succeeded"] == 0.0, "a rank that skipped its step must publish 0"
