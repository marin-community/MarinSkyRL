"""Regression tests for ``RayPPOTrainer._cleanup_old_checkpoints`` dispatch.

The TaskTrove hparam campaign stopped because every arm died at its first
checkpoint (``global_step_3``) with ``WorkerCrashedError``, even though cleanup
was a no-op: ``max_ckpts_to_keep`` resolved to ``-1`` (the default), whose
payload returns immediately. The defect was the unconditional per-node Ray
fan-out in ``run_on_each_node``: on CPU-saturated nodes (Jupiter GH200 with the
agent harness resident) the 0.25-CPU lease with hard affinity failed and
propagated through ``save_checkpoints``, killing runs that had already banked a
checkpoint.

These tests exercise the two contracts the fix restores:

1. When cleanup is disabled, no Ray worker is leased.
2. When the fan-out does fail, it cannot take down a run whose checkpoint is
   already on disk.

    uv run --isolated --extra dev pytest tests/cpu/test_checkpoint_cleanup.py
"""

from unittest.mock import MagicMock, patch

import ray.exceptions

from skyrl_train.trainer import RayPPOTrainer


def _make_bare_trainer(max_ckpts_to_keep: int, node_ids=None) -> RayPPOTrainer:
    """Construct a trainer that only has what ``_cleanup_old_checkpoints`` reads.

    Bypasses ``__init__`` (which builds dataloaders, Ray actor groups, and
    models); the same pattern is used in ``test_resume_overshoot.py``.
    """
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    cfg = MagicMock()
    cfg.trainer.max_ckpts_to_keep = max_ckpts_to_keep
    cfg.trainer.ckpt_path = "/nowhere/unused"
    trainer.cfg = cfg
    trainer._node_ids = node_ids
    trainer.policy_model = MagicMock(name="policy_model")
    trainer.critic_model = None
    trainer.ref_model = None
    return trainer


def test_cleanup_does_not_dispatch_when_disabled():
    """Cleanup must not lease Ray workers when ``max_ckpts_to_keep < 0``.

    This is the regression for the campaign-stopping failure: the default config
    keeps all checkpoints, so the cleanup payload is a no-op, yet the trainer
    still dispatched a fresh 0.25-CPU worker on every node with hard affinity.
    On CPU-saturated nodes the lease failed and the resulting
    ``WorkerCrashedError`` killed runs that had already saved.

    ``run_on_each_node`` is the Ray dispatch boundary -- the thing that leases
    cluster workers -- so asserting it is not called IS the observable contract
    here, not implementation wiring.
    """
    # Three nodes given up front so get_node_ids is not the thing under test.
    trainer = _make_bare_trainer(max_ckpts_to_keep=-1, node_ids=["node-a", "node-b", "node-c"])

    with patch("skyrl_train.trainer.run_on_each_node") as mock_dispatch:
        trainer._cleanup_old_checkpoints()

    mock_dispatch.assert_not_called()


def test_cleanup_isolates_fanout_failure():
    """A fan-out ``WorkerCrashedError`` must not escape ``_cleanup_old_checkpoints``.

    Cleanup runs only after a successful checkpoint save, so it is best-effort
    housekeeping. This is a "does not raise" test because the contract is exactly
    that: a per-node lease failure (e.g. a saturated cluster) is logged and
    swallowed, so ``save_checkpoints`` completes instead of taking the driver
    down. Before the fix, the exception propagated and killed every TaskTrove
    arm at its first checkpoint.
    """
    trainer = _make_bare_trainer(max_ckpts_to_keep=2, node_ids=["node-a", "node-b"])

    with patch("skyrl_train.trainer.run_on_each_node", side_effect=ray.exceptions.WorkerCrashedError):
        # If the regression returns, this raises WorkerCrashedError and fails.
        trainer._cleanup_old_checkpoints()
