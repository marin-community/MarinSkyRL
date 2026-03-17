"""Tests for Best-of-N selection logic."""

import pytest
from skyrl_train.utils.distillation_utils import best_of_n_select


class TestBestOfNSelection:
    """Test best_of_n_select with various scenarios."""

    def test_selects_highest_reward_per_group(self):
        rewards = [0.1, 0.9, 0.5, 0.3, 0.8, 0.2]
        indices = best_of_n_select(rewards, n_samples_per_prompt=3)
        assert len(indices) == 2
        assert rewards[indices[0]] == 0.9
        assert rewards[indices[1]] == 0.8

    def test_n_equals_1(self):
        """With N=1, every sample is selected."""
        rewards = [0.5, 0.3, 0.9, 0.1]
        indices = best_of_n_select(rewards, n_samples_per_prompt=1)
        assert indices == [0, 1, 2, 3]

    def test_n_equals_total(self):
        """With N=total, only one group, best selected."""
        rewards = [0.1, 0.5, 0.9, 0.3]
        indices = best_of_n_select(rewards, n_samples_per_prompt=4)
        assert indices == [2]

    def test_large_n(self):
        rewards = list(range(64))  # 0..63
        indices = best_of_n_select(rewards, n_samples_per_prompt=16)
        assert len(indices) == 4
        # Best in each group of 16: 15, 31, 47, 63
        assert indices == [15, 31, 47, 63]

    def test_negative_rewards(self):
        rewards = [-10, -5, -8, -1, -3, -7]
        indices = best_of_n_select(rewards, n_samples_per_prompt=3)
        assert rewards[indices[0]] == -5
        assert rewards[indices[1]] == -1

    def test_ties_pick_first(self):
        """When rewards are tied, first occurrence should be picked."""
        rewards = [0.5, 0.5, 0.5, 0.5]
        indices = best_of_n_select(rewards, n_samples_per_prompt=2)
        assert len(indices) == 2
        assert indices[0] == 0  # first in group 0
        assert indices[1] == 2  # first in group 1

    def test_mixed_positive_negative(self):
        rewards = [-1.0, 2.0, -0.5, 3.0, -2.0, 1.0]
        indices = best_of_n_select(rewards, n_samples_per_prompt=3)
        assert rewards[indices[0]] == 2.0
        assert rewards[indices[1]] == 3.0

    def test_single_group(self):
        rewards = [0.1, 0.2]
        indices = best_of_n_select(rewards, n_samples_per_prompt=2)
        assert indices == [1]

    def test_many_groups(self):
        # 100 prompts × 4 samples each
        import random
        random.seed(42)
        rewards = [random.random() for _ in range(400)]
        indices = best_of_n_select(rewards, n_samples_per_prompt=4)
        assert len(indices) == 100

        # Verify each index points to the best in its group
        for g, idx in enumerate(indices):
            group_start = g * 4
            group_rewards = rewards[group_start:group_start + 4]
            assert rewards[idx] == max(group_rewards)

    def test_invalid_length_raises(self):
        with pytest.raises(AssertionError, match="divisible"):
            best_of_n_select([0.1, 0.2, 0.3], n_samples_per_prompt=2)

    def test_preserves_absolute_indices(self):
        """Returned indices are absolute (into the flat list), not group-relative."""
        rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        indices = best_of_n_select(rewards, n_samples_per_prompt=3)
        assert indices[0] == 0  # first in group 0 (all tied)
        assert indices[1] == 5  # last in group 1 (only non-zero)
