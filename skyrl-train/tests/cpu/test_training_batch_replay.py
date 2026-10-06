import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import torch
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from tests.training_batch_replay import (
    BatchReplayProvenance,
    CapturingRayPPOTrainer,
    config_fingerprint,
    load_training_batch_artifact,
    replay_policy_forward,
    save_training_batch_artifact,
)


def _batch() -> TrainingInputBatch:
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.bool),
            "response_mask": torch.ones((2, 2), dtype=torch.long),
            "loss_mask": torch.ones((2, 2), dtype=torch.long),
            "rollout_staleness": torch.tensor([0, 1], dtype=torch.int32),
            "rollout_logprobs": None,
            "rollout_routed_experts": torch.arange(24, dtype=torch.int16).reshape(2, 3, 2, 2),
        }
    )
    batch.metadata = {
        "response_length": 2,
        "global_step": 7,
        "uids": ["trial-a", "trial-b"],
        "nested": {"staleness": [0, 1]},
    }
    return batch


def _provenance(**overrides) -> BatchReplayProvenance:
    fields = {
        "source_revision": "a" * 40,
        "config_fingerprint": "b" * 64,
        "checkpoint_path": "/checkpoints/run/global_step_6",
        "checkpoint_step": 6,
        "target_step": 7,
    }
    fields.update(overrides)
    return BatchReplayProvenance(**fields)


class _PolicyForwardBoundary:
    actor_infos = [SimpleNamespace(rank=MeshRank(dp=0, sp=0, tp=0, pp=0, world_size=1, dp_size=1, pp_size=1))]

    def __init__(self, *, fail=False):
        self.fail = fail
        self.drained = False

    def async_run_ray_method(self, dispatch, method, *args, **kwargs):
        if method == "barrier_all":
            self.drained = True
            return [ray.put(None)]
        if method == "empty_cache":
            return []
        if method != "forward":
            raise AssertionError("replay continued past the policy-forward boundary")
        assert self.drained
        if self.fail:
            raise torch.OutOfMemoryError("injected forward OOM")
        batch = kwargs["data"]
        assert batch.metadata["global_step"] == 7
        return [
            ray.put(
                TrainingOutputBatch({"output": torch.full((batch.batch_size, batch.metadata["response_length"]), 0.25)})
            )
        ]


@pytest.fixture
def replay_config():
    cfg = get_default_config()
    cfg.trainer.train_batch_size = 2
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.off_policy_correction = "none"
    return cfg


def test_training_batch_artifact_round_trips_tensors_metadata_and_manifest(tmp_path: Path):
    artifact_path = tmp_path / "step-7-pre-forward"
    original = _batch()
    save_training_batch_artifact(artifact_path, original, _provenance())

    loaded = load_training_batch_artifact(artifact_path, expected=_provenance())

    assert loaded.metadata == original.metadata
    assert loaded.keys() == original.keys()
    for key in original:
        torch.testing.assert_close(loaded[key], original[key], rtol=0, atol=0)

    manifest = json.loads((artifact_path / "manifest.json").read_text())
    assert manifest["tensors"] == {
        "attention_mask": {"dtype": "torch.bool", "shape": [2, 3]},
        "response_mask": {"dtype": "torch.int64", "shape": [2, 2]},
        "loss_mask": {"dtype": "torch.int64", "shape": [2, 2]},
        "rollout_staleness": {"dtype": "torch.int32", "shape": [2]},
        "rollout_routed_experts": {"dtype": "torch.int16", "shape": [2, 3, 2, 2]},
        "sequences": {"dtype": "torch.int64", "shape": [2, 3]},
    }


def test_training_batch_artifact_preserves_compact_router_targets(tmp_path: Path):
    original = _batch()
    dense = original.pop("rollout_routed_experts")
    rows = (dense[0, :2].numpy().copy(), dense[1, :1].numpy().copy())
    original.routed_expert_rows = RoutedExpertRows(rows, response_len=3, num_experts=512)
    artifact_path = tmp_path / "step-7-pre-forward"

    save_training_batch_artifact(artifact_path, original, _provenance())
    restored = load_training_batch_artifact(artifact_path, expected=_provenance())

    expected = torch.zeros((2, 2, 2, 2), dtype=dense.dtype)
    expected[0] = dense[0, :2]
    expected[1, 0] = dense[1, 0]
    torch.testing.assert_close(restored.routed_experts_tensor(), expected)


def test_training_batch_artifact_is_not_published_when_serialization_fails(tmp_path: Path):
    artifact_path = tmp_path / "step-7-pre-forward"
    batch = _batch()
    batch.metadata["not_pickleable"] = lambda: None

    with pytest.raises(Exception):
        save_training_batch_artifact(artifact_path, batch, _provenance())

    assert not artifact_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_training_batch_artifact_rejects_a_modified_payload(tmp_path: Path):
    artifact_path = tmp_path / "step-7-pre-forward"
    save_training_batch_artifact(artifact_path, _batch(), _provenance())
    with (artifact_path / "batch.pkl").open("ab") as file:
        file.write(b"modified")

    with pytest.raises(ValueError, match="batch_sha256 mismatch"):
        load_training_batch_artifact(artifact_path, expected=_provenance())


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("source_revision", "c" * 40),
        ("config_fingerprint", "d" * 64),
        ("checkpoint_path", "/checkpoints/other/global_step_6"),
        ("checkpoint_step", 5),
        ("target_step", 8),
    ],
)
def test_replay_rejects_provenance_mismatch(tmp_path: Path, field: str, wrong_value):
    artifact_path = tmp_path / "step-7-pre-forward"
    save_training_batch_artifact(artifact_path, _batch(), _provenance())

    with pytest.raises(ValueError, match=field):
        load_training_batch_artifact(artifact_path, expected=_provenance(**{field: wrong_value}))


def test_config_fingerprint_excludes_only_the_diagnostic_controls():
    capture = OmegaConf.create(
        {
            "trainer": {"policy": {"model": {"path": "model"}}, "train_batch_size": 8},
            "batch_replay": {"mode": "capture", "artifact_path": "/first"},
        }
    )
    replay = copy.deepcopy(capture)
    replay.batch_replay.mode = "replay"
    replay.batch_replay.artifact_path = "/second"

    assert config_fingerprint(capture) == config_fingerprint(replay)

    replay.trainer.train_batch_size = 16
    assert config_fingerprint(capture) != config_fingerprint(replay)


@pytest.mark.asyncio
async def test_capture_survives_a_subsequent_forward_failure(tmp_path: Path, replay_config, driver_trainer_factory):
    trainer = driver_trainer_factory(
        replay_config,
        trainer_type=CapturingRayPPOTrainer,
        capture_artifact_path=tmp_path / "step-7-pre-forward",
        capture_provenance=_provenance(),
    )
    trainer.global_step = 7
    trainer.policy_model = _PolicyForwardBoundary(fail=True)
    with pytest.raises(torch.OutOfMemoryError, match="injected forward OOM"):
        await replay_policy_forward(trainer, _batch())
    restored = load_training_batch_artifact(
        trainer.capture_artifact_path,
        expected=trainer.capture_provenance,
    )
    torch.testing.assert_close(restored["sequences"], _batch()["sequences"])


@pytest.mark.asyncio
async def test_replay_uses_production_step_prefix_and_stops_after_forward(replay_config, driver_trainer_factory):
    trainer = driver_trainer_factory(replay_config)
    trainer.global_step = 7
    trainer.policy_model = _PolicyForwardBoundary()
    batch = _batch()
    del batch.metadata["global_step"]
    result = await replay_policy_forward(trainer, batch)
    torch.testing.assert_close(result["action_log_probs"], torch.full((2, 2), 0.25))
    assert "advantages" not in result
