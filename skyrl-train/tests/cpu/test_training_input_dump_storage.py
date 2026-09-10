"""Training-input dumps preserve tensors and metadata on local and object paths."""

import pickle
from types import SimpleNamespace

import fsspec
import pytest
import torch

from skyrl_train.io import io
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch


@pytest.mark.parametrize("remote", [False, True])
def test_actual_dump_roundtrips_prepared_inputs_at_requested_storage_root(tmp_path, monkeypatch, remote):
    monkeypatch.chdir(tmp_path)
    memory = fsspec.filesystem("memory")
    opened = []

    class ObjectStore:
        @staticmethod
        def _strip_protocol(path):
            return path.removeprefix("s3://")

        def open(self, path, mode):
            opened.append((path, mode))
            return memory.open(path, mode)

    if remote:
        monkeypatch.setattr(io, "_get_filesystem", lambda _path: ObjectStore())
    root = "s3://n2-input-dump-test/exports" if remote else str(tmp_path / "exports")
    trainer = SimpleNamespace(cfg=SimpleNamespace(trainer=SimpleNamespace(export_path=root)))
    batch = TrainingInputBatch(
        {
            "action_log_probs": torch.tensor([[-0.2, -0.3], [-0.4, 0.0]]),
            "advantages": torch.tensor([[1.0, 1.0], [-1.0, 0.0]]),
            "loss_mask": torch.tensor([[1, 1], [1, 0]]),
        }
    )
    batch.metadata = {"uids": ["first", "second"], "async_cohort_admission_step": 7, "async_cohort_update_index": 1}
    RayPPOTrainer.dump_data(trainer, batch, "global_step_8_training_input")
    path = root + "/dumped_data/global_step_8_training_input.pkl"
    if remote:
        assert opened == [(path.removeprefix("s3://"), "wb")]
        payload = memory.cat(path.removeprefix("s3://"))
        assert not (tmp_path / "s3:").exists()
    else:
        payload = (tmp_path / "exports/dumped_data/global_step_8_training_input.pkl").read_bytes()
    restored = pickle.loads(payload)
    assert isinstance(restored, TrainingInputBatch)
    assert restored.metadata == batch.metadata
    assert restored.keys() == batch.keys()
    for key in batch:
        assert torch.equal(restored[key], batch[key])
