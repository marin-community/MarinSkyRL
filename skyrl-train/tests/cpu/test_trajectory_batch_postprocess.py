import pytest
from skyrl_gym.verification import VerificationResult

from skyrl_train.trajectory_runners.base import TrajectoryBatch, propagate_data_sources
from skyrl_train.trajectory_runners.rollout_metrics import _domain_reward_metrics, observe_rollout, reward_metrics
from skyrl_train.trajectory_runners.trajectory_processing import token_reward_rows
from skyrl_train.trajectory_runners.types import BatchFields, TrajectoryID


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        # Response-level rewards are credited to the final response token.
        ([1.0, 0.5], [[0.0, 1.0], [0.0, 0.0, 0.5]]),
        # Token-level rewards pass through unchanged.
        ([[0.1, 0.3], [0.2, 0.1, 0.1]], [[0.1, 0.3], [0.2, 0.1, 0.1]]),
    ],
)
def test_reward_rows_produce_per_token_rewards(rewards, expected):
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": rewards,
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = token_reward_rows(trajectory_batch["rewards"], trajectory_batch["response_ids"])

    assert result == expected


def test_pass_at_k_uses_unshaped_outcomes():
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [1], [2], [2]],
        "response_ids": [[3], [4], [5], [6]],
        "rewards": [0.2, 0.3, 0.4, 0.5],
        "unshaped_rewards": [0.0, 1.0, 0.0, 0.0],
        "loss_masks": [[1], [1], [1], [1]],
        "stop_reasons": ["stop", "stop", "stop", "stop"],
        "rollout_metrics": None,
    }

    observations = observe_rollout(trajectory_batch, fields=BatchFields.from_batch(trajectory_batch))
    metrics = reward_metrics(observations, ["a", "a", "b", "b"], n_samples_per_prompt=2, step_wise=False)

    assert metrics["reward/avg_pass_at_2"] == 0.5
    assert metrics["reward/avg_raw_reward"] == pytest.approx(0.35)


def test_training_reports_normalized_composite_scores_by_agent():
    batch: TrajectoryBatch = {
        "response_ids": [[1], [2], [3]],
        "rewards": [5.0, 1.0, 0.0],
        "verification_results": [
            VerificationResult.verified(5.0, diagnostics={"agent": "genrm"}, score_min=1.0, score_max=5.0),
            VerificationResult.verified(1.0, diagnostics={"agent": "mcqa"}),
            VerificationResult.error("judge unavailable", diagnostics={"agent": "mcqa"}),
        ],
    }

    observations = observe_rollout(batch, fields=BatchFields.from_batch(batch))
    metrics = reward_metrics(observations, ["a", "b", "c"], n_samples_per_prompt=1, step_wise=False)

    assert metrics["reward/avg_raw_reward"] == 2.0
    assert metrics["reward/avg_verifier_score"] == pytest.approx(2 / 3)
    assert metrics["reward/agent/genrm/avg_verifier_score"] == 1.0
    assert metrics["reward/agent/mcqa/avg_verifier_score"] == 0.5


def test_informative_group_fraction_counts_groups_whose_rewards_differ():
    # Group a has reward spread; group b is a tie and carries no advantage signal.
    batch = {"response_ids": [[1], [2], [3], [4]], "rewards": [1.0, 0.0, 0.5, 0.5]}
    observations = observe_rollout(batch, fields=BatchFields.from_batch(batch))
    metrics = reward_metrics(observations, ["a", "a", "b", "b"], n_samples_per_prompt=1, step_wise=False)
    assert metrics["reward/informative_group_fraction"] == 0.5


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


def test_reward_metrics_leave_out_skipped_rollouts():
    skipped = VerificationResult.skipped("grading is skipped")
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [1], [2], [2]],
        "response_ids": [[3], [4], [5], [6]],
        "rewards": [1.0, 0.0, 0.0, 0.0],
        "verification_results": [
            VerificationResult.verified(1.0, passed=True),
            VerificationResult.verified(0.0, passed=False),
            skipped,
            skipped,
        ],
        "data_sources": ["lean", "lean", "ultra", "ultra"],
        "loss_masks": [[1], [1], [1], [1]],
        "rollout_metrics": None,
    }

    observations = observe_rollout(trajectory_batch, fields=BatchFields.from_batch(trajectory_batch))
    metrics = reward_metrics(observations, ["a", "a", "b", "b"], n_samples_per_prompt=2, step_wise=False)

    assert metrics["reward/avg_raw_reward"] == 0.5
    assert metrics["reward/avg_pass_at_2"] == 1.0
    assert metrics["reward/informative_group_fraction"] == 1.0
    assert metrics["reward/domain/lean/avg_raw_reward"] == 0.5
    assert "reward/domain/ultra/avg_raw_reward" not in metrics


def test_all_skipped_rollouts_record_no_reward_metrics():
    skipped = VerificationResult.skipped("grading is skipped")
    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[1], [2]],
        "response_ids": [[3], [4]],
        "rewards": [0.0, 0.0],
        "verification_results": [skipped, skipped],
        "loss_masks": [[1], [1]],
        "rollout_metrics": None,
    }

    observations = observe_rollout(trajectory_batch, fields=BatchFields.from_batch(trajectory_batch))
    metrics = reward_metrics(observations, ["a", "b"], n_samples_per_prompt=1, step_wise=False)

    assert not any(key.startswith("reward/") for key in metrics)
    assert token_reward_rows(trajectory_batch["rewards"], trajectory_batch["response_ids"]) == [[0.0], [0.0]]
