"""Tests for shared Best-of-N trajectory selection."""

import pytest
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector, best_of_n_indices
from skyrl_train.trajectory_runners.types import TrajectoryID


class TestBestOfNSelection:
    """Test best_of_n_indices with various scenarios."""

    def test_selects_highest_reward_per_group(self):
        rewards = [0.1, 0.9, 0.5, 0.3, 0.8, 0.2]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=3)
        assert len(indices) == 2
        assert rewards[indices[0]] == 0.9
        assert rewards[indices[1]] == 0.8

    def test_n_equals_1(self):
        """With N=1, every sample is selected."""
        rewards = [0.5, 0.3, 0.9, 0.1]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=1)
        assert indices == [0, 1, 2, 3]

    def test_n_equals_total(self):
        """With N=total, only one group, best selected."""
        rewards = [0.1, 0.5, 0.9, 0.3]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=4)
        assert indices == [2]

    def test_large_n(self):
        rewards = list(range(64))  # 0..63
        indices = best_of_n_indices(rewards, n_samples_per_prompt=16)
        assert len(indices) == 4
        # Best in each group of 16: 15, 31, 47, 63
        assert indices == [15, 31, 47, 63]

    def test_negative_rewards(self):
        rewards = [-10, -5, -8, -1, -3, -7]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=3)
        assert rewards[indices[0]] == -5
        assert rewards[indices[1]] == -1

    def test_ties_pick_first(self):
        """When rewards are tied, first occurrence should be picked."""
        rewards = [0.5, 0.5, 0.5, 0.5]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=2)
        assert len(indices) == 2
        assert indices[0] == 0  # first in group 0
        assert indices[1] == 2  # first in group 1

    def test_mixed_positive_negative(self):
        rewards = [-1.0, 2.0, -0.5, 3.0, -2.0, 1.0]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=3)
        assert rewards[indices[0]] == 2.0
        assert rewards[indices[1]] == 3.0

    def test_single_group(self):
        rewards = [0.1, 0.2]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=2)
        assert indices == [1]

    def test_many_groups(self):
        # 100 prompts × 4 samples each
        import random

        random.seed(42)
        rewards = [random.random() for _ in range(400)]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=4)
        assert len(indices) == 100

        # Verify each index points to the best in its group
        for g, idx in enumerate(indices):
            group_start = g * 4
            group_rewards = rewards[group_start : group_start + 4]
            assert rewards[idx] == max(group_rewards)

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            best_of_n_indices([0.1, 0.2, 0.3], n_samples_per_prompt=2)

    def test_preserves_absolute_indices(self):
        """Returned indices are absolute (into the flat list), not group-relative."""
        rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        indices = best_of_n_indices(rewards, n_samples_per_prompt=3)
        assert indices[0] == 0  # first in group 0 (all tied)
        assert indices[1] == 5  # last in group 1 (only non-zero)


def test_best_of_n_selector_filters_every_trajectory_channel_and_uid_before_teacher_scoring():
    selector = BestOfNTrajectorySelector(n_samples_per_prompt=2)
    trajectory_batch = {
        "prompt_token_ids": [[10], [10], [20], [20]],
        "response_ids": [[1], [2], [3], [4]],
        "rewards": [[0.1], [0.9], [-0.2], [-0.1]],
        "loss_masks": [[1], [1], [1], [1]],
        "trajectory_ids": [
            TrajectoryID("a", 0),
            TrajectoryID("a", 1),
            TrajectoryID("b", 0),
            TrajectoryID("b", 1),
        ],
        "stop_reasons": ["stop-0", "stop-1", "stop-2", "stop-3"],
        "rollout_metrics": {"generation/count": 4},
    }

    selection = selector.select(trajectory_batch, ["a", "a", "b", "b"])

    assert selection.trajectory_batch["response_ids"] == [[2], [4]]
    assert selection.trajectory_batch["trajectory_ids"] == [TrajectoryID("a", 1), TrajectoryID("b", 1)]
    assert selection.trajectory_batch["stop_reasons"] == ["stop-1", "stop-3"]
    assert selection.trajectory_batch["rollout_metrics"] == {"generation/count": 4}
    assert selection.uids == ["a", "b"]
    assert selection.metrics == {
        "best_of_n/best_reward_mean": pytest.approx(0.4),
        "best_of_n/group_reward_mean": pytest.approx(0.175),
        "best_of_n/reward_improvement": pytest.approx(0.225),
        "best_of_n/n_samples_per_prompt": 2.0,
    }
