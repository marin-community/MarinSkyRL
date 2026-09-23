"""Tests for ``RayPPOTrainer._cleanup_old_checkpoints`` dispatch behavior.

Contracts covered: cleanup leases no Ray worker when disabled
(``max_ckpts_to_keep < 0``); and a per-node dispatch failure cannot kill a run
whose checkpoint is already on disk, nor block the driver-side cleanup pass.

Background on the incident that motivated these tests is in
``docs/debug-log-checkpoint-cleanup-fanout.md``.

    uv run --isolated --extra dev pytest tests/cpu/test_checkpoint_cleanup.py
"""

from unittest.mock import MagicMock, patch

import ray.exceptions

from skyrl_train.trainer import RayPPOTrainer


def _make_bare_trainer(max_ckpts_to_keep: int, ckpt_path: str, node_ids) -> RayPPOTrainer:
    """Return a trainer with only the fields ``_cleanup_old_checkpoints`` reads."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    cfg = MagicMock()
    cfg.trainer.max_ckpts_to_keep = max_ckpts_to_keep
    cfg.trainer.ckpt_path = ckpt_path
    trainer.cfg = cfg
    trainer._node_ids = node_ids
    return trainer


def test_cleanup_takes_no_cluster_dependency_when_disabled():
    """Cleanup with ``max_ckpts_to_keep < 0`` does not lease a Ray worker.

    The payload is a no-op at this value, so the only thing the fan-out can do is
    fail; ``run_on_each_node`` is the Ray lease boundary, so its non-invocation
    is the contract under test. The failure-isolation test cannot substitute:
    once the fan-out is wrapped in ``except RayError``, a "does it raise" check
    passes whether or not this early return exists.
    """
    trainer = _make_bare_trainer(max_ckpts_to_keep=-1, ckpt_path="/unused", node_ids=["a", "b", "c"])

    with patch("skyrl_train.trainer.run_on_each_node") as mock_dispatch:
        trainer._cleanup_old_checkpoints()

    mock_dispatch.assert_not_called()


def test_cleanup_runs_driver_side_after_fanout_failure(tmp_path):
    """A per-node dispatch failure must not kill the run or skip driver cleanup.

    Cleanup runs after a successful checkpoint save, so it is best-effort: the
    Ray failure is swallowed and the independent driver-side pass still removes
    old checkpoints so a shared ``ckpt_path`` does not accumulate.
    """
    for step in (1, 2, 3):
        checkpoint_dir = tmp_path / f"global_step_{step}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "trainer_state.pt").write_bytes(b"legacy checkpoint")
    trainer = _make_bare_trainer(max_ckpts_to_keep=1, ckpt_path=str(tmp_path), node_ids=["a", "b"])

    with patch("skyrl_train.trainer.run_on_each_node", side_effect=ray.exceptions.WorkerCrashedError):
        trainer._cleanup_old_checkpoints()

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["global_step_3"], "driver-side cleanup should keep only the newest checkpoint"


def test_cloud_cleanup_runs_once_on_driver_without_node_leases():
    checkpoint_root = "s3://bucket/checkpoints"
    trainer = _make_bare_trainer(max_ckpts_to_keep=2, ckpt_path=checkpoint_root, node_ids=[])

    with (
        patch("skyrl_train.trainer.protected_hf_export_steps", return_value={3}),
        patch("skyrl_train.trainer.get_node_ids") as mock_node_ids,
        patch("skyrl_train.trainer.run_on_each_node") as mock_dispatch,
        patch("skyrl_train.trainer.cleanup_old_checkpoints") as mock_cleanup,
    ):
        trainer._cleanup_old_checkpoints()

    mock_node_ids.assert_not_called()
    mock_dispatch.assert_not_called()
    mock_cleanup.assert_called_once_with(checkpoint_root, 2, {3})


def test_driver_retention_zero_still_keeps_advertised_latest():
    checkpoint_root = "s3://bucket/checkpoints"
    trainer = _make_bare_trainer(max_ckpts_to_keep=0, ckpt_path=checkpoint_root, node_ids=[])

    with (
        patch("skyrl_train.trainer.protected_hf_export_steps", return_value=set()),
        patch("skyrl_train.trainer.cleanup_old_checkpoints") as mock_cleanup,
    ):
        trainer._cleanup_old_checkpoints()

    mock_cleanup.assert_called_once_with(checkpoint_root, 1, set())
