"""
Test for token-level rewards support in RayPPOTrainer.postprocess_trajectory_batch method.

Run with:
uv run --isolated --group dev --extra cpu pytest tests/cpu/test_trajectory_batch_postprocess.py
"""

from unittest.mock import MagicMock

import pytest

from skyrl_train.trainer import RayPPOTrainer, _domain_reward_metrics
from skyrl_train.trajectory_runners.base import TrajectoryBatch, propagate_data_sources
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.config.utils import get_default_config
from omegaconf import OmegaConf


class DummyDataset:
    def __len__(self):
        return 1

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


def test_response_level_rewards():
    """Test postprocess_trajectory_batch with response-level rewards (List[float])."""

    # Test length=1
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
    )

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
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
    )

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
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
    )

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
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
    )

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
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
    )
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


def test_domain_reward_metrics_aggregate_and_bound_metric_keys():
    metrics = _domain_reward_metrics(["alpha", "alpha", None, "zeta"], [0.0, 1.0, 0.5, 0.0])
    assert metrics == {
        "reward/domain/alpha/avg_raw_reward": 0.5,
        "reward/domain/_missing/avg_raw_reward": 0.5,
        "reward/domain/zeta/avg_raw_reward": 0.0,
    }

    overflow_metrics = _domain_reward_metrics(["__other__", *[f"domain-{i:02}" for i in range(40)]], [0.0] + [1.0] * 40)
    assert len(overflow_metrics) == 33
    assert overflow_metrics["reward/domain/__other__/avg_raw_reward"] == 0.0
    assert overflow_metrics["reward/domain_overflow/avg_raw_reward"] == 1.0


def test_domain_reward_metric_names_do_not_merge_distinct_sources():
    metrics = _domain_reward_metrics(
        ["a/b", "a_b", None, "unknown", "_missing", "_source_a_2fb", "Math", "math"],
        [0.0, 1.0, 0.25, 0.75, 0.5, 0.6, 0.3, 0.9],
    )

    assert metrics == {
        "reward/domain/_source_a_2fb/avg_raw_reward": 0.0,
        "reward/domain/a_b/avg_raw_reward": 1.0,
        "reward/domain/_missing/avg_raw_reward": 0.25,
        "reward/domain/unknown/avg_raw_reward": 0.75,
        "reward/domain/_source__5fmissing/avg_raw_reward": 0.5,
        "reward/domain/_source__5fsource_5fa_5f2fb/avg_raw_reward": 0.6,
        "reward/domain/_source__4dath/avg_raw_reward": 0.3,
        "reward/domain/math/avg_raw_reward": 0.9,
    }


def test_step_wise_rollout_rows_keep_request_data_sources():
    request = {
        "env_extras": [{"extra_info": {"data_source": "math"}}, {"data_source": "tools"}],
        "trajectory_ids": [TrajectoryID("a", 0), TrajectoryID("b", 0)],
    }
    output = {
        "response_ids": [[1], [2], [3]],
        "trajectory_ids": [TrajectoryID("a", 0), TrajectoryID("a", 0), TrajectoryID("b", 0)],
    }

    propagate_data_sources(request, output)

    assert output["data_sources"] == ["math", "math", "tools"]

    output_with_empty_first_trajectory = {
        "response_ids": [[1], [2]],
        "trajectory_ids": [TrajectoryID("b", 0), TrajectoryID("b", 0)],
    }
    propagate_data_sources(request, output_with_empty_first_trajectory)
    assert output_with_empty_first_trajectory["data_sources"] == ["tools", "tools"]


def test_postprocess_logs_training_reward_by_domain():
    trainer = RayPPOTrainer(
        cfg=create_config(2),
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        trajectory_runner=MagicMock(),
    )
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [2]],
        "response_ids": [[3], [4]],
        "rewards": [0.8, 0.6],
        "unshaped_rewards": [0.0, 1.0],
        "data_sources": ["math", "tools"],
        "loss_masks": [[1], [1]],
        "rollout_metrics": None,
    }

    trainer.postprocess_trajectory_batch(trajectory_batch, ["a", "b"])

    assert trainer.all_metrics["reward/domain/math/avg_raw_reward"] == 0.8
    assert trainer.all_metrics["reward/domain/tools/avg_raw_reward"] == 0.6
