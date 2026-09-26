"""
Test for token-level rewards support in RayPPOTrainer.postprocess_trajectory_batch method.

Run with:
uv run --isolated --group dev --extra cpu pytest tests/cpu/test_trajectory_batch_postprocess.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.base import TrajectoryBatch
from skyrl_train.config.utils import get_default_config
from omegaconf import OmegaConf


class DummyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def create_config(batch_size):
    default_config = get_default_config()
    OmegaConf.update(
        default_config,
        "trainer",
        {
            "train_batch_size": batch_size,
            "eval_batch_size": batch_size,
            "resume_mode": "none",
            "seed": 42,
            "epochs": 1,
        },
    )
    OmegaConf.update(
        default_config,
        "generator",
        {
            "n_samples_per_prompt": 1,
        },
    )
    return default_config


def make_trainer(config) -> RayPPOTrainer:
    batch_size = config.trainer.train_batch_size
    return RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
        context=SimpleNamespace(config=SimpleNamespace(max_staleness_steps=0, batch_size=batch_size)),
    )


def test_response_level_rewards():
    """Test postprocess_trajectory_batch with response-level rewards (List[float])."""

    # Test length=1
    trainer = make_trainer(create_config(1))

    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": [1.0],  # Response-level reward
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_trajectory_batch(trajectory_batch, ["uid1"])

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 0.0, 1.0]]

    # Test length=2
    trainer = make_trainer(create_config(2))

    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": [1.0, 0.5],  # Response-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_trajectory_batch(trajectory_batch, ["uid1", "uid2"])

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 1.0], [0.0, 0.0, 0.5]]


def test_token_level_rewards():
    """Test postprocess_trajectory_batch with token-level rewards (List[List[float]])."""

    # Test length=1
    trainer = make_trainer(create_config(1))

    per_token_rewards = [[0.1, 0.2, 0.3]]
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_trajectory_batch(trajectory_batch, ["uid1"])

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards

    # Test length=2
    trainer = make_trainer(create_config(2))

    per_token_rewards = [[0.1, 0.3], [0.2, 0.1, 0.1]]
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_trajectory_batch(trajectory_batch, ["uid1", "uid2"])

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards


def test_pass_at_k_uses_unshaped_outcomes():
    config = create_config(4)
    config.generator.n_samples_per_prompt = 2
    trainer = make_trainer(config)
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [1], [2], [2]],
        "response_ids": [[3], [4], [5], [6]],
        "rewards": [0.2, 0.3, 0.4, 0.5],
        "unshaped_rewards": [0.0, 1.0, 0.0, 0.0],
        "loss_masks": [[1], [1], [1], [1]],
        "stop_reasons": ["stop", "stop", "stop", "stop"],
        "rollout_metrics": None,
    }

    trainer.postprocess_trajectory_batch(trajectory_batch, ["a", "a", "b", "b"])

    assert trainer.all_metrics["reward/avg_pass_at_2"] == 0.5
    assert trainer.all_metrics["reward/avg_raw_reward"] == pytest.approx(0.35)
