"""Tests for shared Best-of-N trajectory selection."""

import pytest
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector, best_of_n_indices
from skyrl_train.trajectory_runners.types import TrajectoryID


@pytest.mark.parametrize(
    ("rewards", "n_samples_per_prompt", "expected"),
    [
        pytest.param([0.1, 0.9, 0.5, 0.3, 0.8, 0.2], 3, [1, 4], id="argmax-per-group"),
        pytest.param([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3, [0, 5], id="absolute-indices"),
        pytest.param([0.5, 0.5, 0.5, 0.5], 2, [0, 2], id="ties-pick-first"),
        pytest.param([0.5, 0.3, 0.9, 0.1], 1, [0, 1, 2, 3], id="n-equals-1"),
    ],
)
def test_best_of_n_indices_selects_first_best_sample_per_group(rewards, n_samples_per_prompt, expected):
    assert best_of_n_indices(rewards, n_samples_per_prompt=n_samples_per_prompt) == expected


def test_best_of_n_indices_rejects_partial_group():
    with pytest.raises(ValueError, match="divisible"):
        best_of_n_indices([0.1, 0.2, 0.3], n_samples_per_prompt=2)


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
