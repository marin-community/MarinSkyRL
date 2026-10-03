from unittest.mock import MagicMock, patch

import ray.exceptions
import pytest

from marinskyrl.checkpoint_paths import LATEST_CHECKPOINT_FILE
from skyrl_train.checkpoint_generation import commit_attempt, new_attempt_path, resolve_checkpoint_payload
from skyrl_train.hf_export import write_hf_export_request
from skyrl_train.hf_export_schema import HFExportRequest, TRAINER_STATE_FILENAME
from skyrl_train.io import io
from skyrl_train.trainer import RayPPOTrainer
from tests.cpu.utils.test_io import GcsMemoryFileSystem


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
    trainer = _make_bare_trainer(max_ckpts_to_keep=-1, ckpt_path="/unused", node_ids=["a", "b", "c"])

    with patch("skyrl_train.trainer.run_on_each_node") as mock_dispatch:
        trainer._cleanup_old_checkpoints()

    mock_dispatch.assert_not_called()


def test_cleanup_runs_driver_side_after_fanout_failure(tmp_path):
    for step in (1, 2, 3):
        checkpoint_dir = tmp_path / f"global_step_{step}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "trainer_state.pt").write_bytes(b"legacy checkpoint")
    trainer = _make_bare_trainer(max_ckpts_to_keep=1, ckpt_path=str(tmp_path), node_ids=["a", "b"])

    with patch("skyrl_train.trainer.run_on_each_node", side_effect=ray.exceptions.WorkerCrashedError):
        trainer._cleanup_old_checkpoints()

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["global_step_3"], "driver-side cleanup should keep only the newest checkpoint"


@pytest.mark.parametrize("retention", [0, 1])
def test_cloud_retention_keeps_latest_and_pending_export_without_cluster_access(monkeypatch, retention):
    filesystem = GcsMemoryFileSystem()
    monkeypatch.setattr(io, "_get_filesystem", lambda path: filesystem)
    checkpoint_root = "gs://bucket/checkpoints"
    for step in (1, 2, 3, 4):
        step_path = f"{checkpoint_root}/global_step_{step}"
        attempt = new_attempt_path(step_path)
        io.write_bytes_atomic(f"{attempt}/{TRAINER_STATE_FILENAME}", b"trainer state")
        commit_attempt(step_path, attempt, required_files={TRAINER_STATE_FILENAME})
    failed_attempt = new_attempt_path(f"{checkpoint_root}/global_step_0")
    io.write_bytes_atomic(f"{failed_attempt}/{TRAINER_STATE_FILENAME}", b"uncommitted state")
    io.write_bytes_atomic(f"{checkpoint_root}/{LATEST_CHECKPOINT_FILE}", b"2")
    write_hf_export_request(
        HFExportRequest(
            step=1,
            checkpoint_base_path=checkpoint_root,
            checkpoint_path=f"{checkpoint_root}/global_step_1",
            export_path="gs://bucket/exports",
            model_path="org/model",
            num_nodes=1,
            gpus_per_node=1,
        )
    )
    trainer = _make_bare_trainer(retention, checkpoint_root, node_ids=[])
    trainer._cleanup_old_checkpoints()

    assert not io.exists(f"{checkpoint_root}/global_step_3")
    assert not io.exists(f"{checkpoint_root}/global_step_0")
    for step in (1, 2, 4):
        assert resolve_checkpoint_payload(f"{checkpoint_root}/global_step_{step}", verify_files=True)
