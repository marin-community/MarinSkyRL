from skyrl_train.batch_sampling import filter_trajectory_batch
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from dataclasses import replace

import numpy as np
from omegaconf import OmegaConf
from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult

from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.trajectory_runners.types import AgentLoopOutput, TrajectoryID


class _Tokenizer:
    eos_token_id = 99


def _config():
    return OmegaConf.create(
        {
            "apply_overlong_filtering": False,
            "sampling_params": {"logprobs": True},
        }
    )


def _step(response_ids, reward, *, token_provenance="engine"):
    outcome = float(sum(reward) if isinstance(reward, list) else reward)
    return AgentLoopOutput(
        evidence=RolloutEvidence(
            stop_reason="stop",
            prompt_token_ids=(1, 2),
            response_token_ids=tuple(response_ids),
            behavior_logprobs=np.full(len(response_ids), -0.1, dtype=np.float32),
        ),
        verification=VerificationResult.verified(outcome),
        reward=RewardResult(
            unshaped_reward=outcome,
            optimization_reward=outcome,
            token_rewards=tuple(reward) if isinstance(reward, list) else None,
        ),
        disposition=TrainingDisposition.train(),
        loss_mask=[1] * len(response_ids),
        env_metrics={"score": outcome},
        token_provenance=token_provenance,
    )


def test_whole_trajectory_projection_preserves_one_sample_per_trajectory():
    projection = WholeTrajectoryProjection(_config(), _Tokenizer())
    output = projection.project(
        [_step([3, 4], [0.0, 1.0])],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )

    assert output["response_ids"] == [[3, 4]]
    assert output["rewards"] == [[0.0, 1.0]]
    assert output["loss_masks"] == [[1, 1]]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.1, -0.1])
    assert output["rollout_metrics"]["generate/token_provenance/reconstructed_fraction"] == 0.0
    assert "trajectory_ids" not in output


def test_whole_trajectory_projection_preserves_partial_server_error_diagnostics():
    failed = replace(
        _step([3], 0.0),
        verification=VerificationResult.error(
            "model server rejected generation", diagnostics={"error_category": "constrained_decoding"}
        ),
        disposition=TrainingDisposition.mask("model server error", exception_type="ModelServerError"),
    )

    output = WholeTrajectoryProjection(_config(), _Tokenizer()).project(
        [failed], {"env_classes": None, "sampling_params": {"logprobs": True}}
    )

    assert output["loss_masks"] == [[0]]
    assert output["server_errors"] == [{"category": "constrained_decoding", "request_id": None, "status_code": None}]


def test_whole_trajectory_projection_preserves_routes_and_fills_missing_rows():
    routed = _step([3, 4], [0.0, 1.0])
    routed.evidence = replace(routed.evidence, routed_experts=np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8))

    output = WholeTrajectoryProjection(_config(), _Tokenizer()).project(
        [routed, _step([5], [0.0])],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )

    np.testing.assert_array_equal(output["rollout_routed_experts"][0], [[[1, 2]], [[3, 4]]])
    np.testing.assert_array_equal(output["rollout_routed_experts"][1], [[[0, 0]]])


def test_step_wise_projection_preserves_group_identity_and_final_step():
    projection = StepWiseTrajectoryProjection(_config(), _Tokenizer())
    trajectory_id = TrajectoryID(instance_id="task", repetition_id=2)
    output = projection.project(
        [
            [
                _step([3], [1.0]),
                _step([4, 5], [0.0, 2.0], token_provenance="reconstructed"),
            ]
        ],
        {
            "env_classes": ["math"],
            "trajectory_ids": [trajectory_id],
            "sampling_params": {"logprobs": True},
        },
    )

    assert output["response_ids"] == [[3], [4, 5]]
    assert output["rewards"] == [[1.0], [0.0, 2.0]]
    assert output["loss_masks"] == [[1], [1, 1]]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.1])
    np.testing.assert_allclose(output["rollout_logprobs"][1], [-0.1, -0.1])
    assert [(item.instance_id, item.repetition_id, item.step) for item in output["trajectory_ids"]] == [
        ("task", 2, 0),
        ("task", 2, 1),
    ]
    assert output["is_last_step"] == [False, True]
    assert output["rollout_metrics"]["generate/token_provenance/reconstructed_fraction"] == 0.5


def test_step_wise_projection_preserves_routes():
    step = _step([3, 4], [0.0, 1.0])
    step.evidence = replace(step.evidence, routed_experts=np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8))

    output = StepWiseTrajectoryProjection(_config(), _Tokenizer()).project(
        [[step]],
        {
            "env_classes": ["math"],
            "trajectory_ids": [TrajectoryID("task", 0)],
            "sampling_params": {"logprobs": True},
        },
    )

    np.testing.assert_array_equal(output["rollout_routed_experts"][0], [[[1, 2]], [[3, 4]]])


def test_step_wise_projection_preserves_student_topk_candidates():
    projection = StepWiseTrajectoryProjection(_config(), _Tokenizer())
    first = _step([3], 1.0)
    first.evidence = replace(
        first.evidence,
        student_topk_indices=np.asarray([[3, 4]], dtype=np.int32),
        behavior_topk_logprobs=np.asarray([[-0.1, -1.1]], dtype=np.float32),
    )
    second = _step([5], 0.0)
    second.evidence = replace(
        second.evidence,
        student_topk_indices=np.asarray([[5, 6]], dtype=np.int32),
        behavior_topk_logprobs=np.asarray([[-0.2, -1.2]], dtype=np.float32),
    )

    output = projection.project(
        [[first, second]],
        {"env_classes": ["math"], "trajectory_ids": [TrajectoryID("task", 0)], "sampling_params": {"logprobs": 2}},
    )

    np.testing.assert_array_equal(output["student_topk_indices"][0], [[3, 4]])
    np.testing.assert_array_equal(output["student_topk_indices"][1], [[5, 6]])
    np.testing.assert_allclose(output["behavior_topk_logprobs"][0], [[-0.1, -1.1]])
    np.testing.assert_allclose(output["behavior_topk_logprobs"][1], [[-0.2, -1.2]])


def test_projection_derives_mask_baseline_and_token_credit_from_contracts():
    projection = WholeTrajectoryProjection(_config(), _Tokenizer())
    step = _step([3, 4], 0.0)
    step = replace(
        step,
        reward=RewardResult(unshaped_reward=None, optimization_reward=0.0, token_credit=(0.1, -0.1)),
        disposition=TrainingDisposition.mask("verifier unavailable", exception_type="TurnCapExhaustedError"),
    )
    step.error_treatment = "passthrough"

    output = projection.project(
        [step],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )

    assert output["loss_masks"] == [[0, 0]]
    assert output["exclude_from_baseline"] == [True]
    assert output["token_level_shaping"] == [[0.1, -0.1]]
    assert output["exception_types"] == ["TurnCapExhaustedError"]
    assert output["error_treatments"] == ["passthrough"]
    assert output["unshaped_rewards"] == [0.0]
    assert output["unshaped_reward_available"] == [False]


def test_whole_trajectory_projection_carries_environment_rates_into_async_batch():
    n10_hit = replace(_step([3], 1.0), env_metrics={"exact_n10": 1.0})
    n10_miss = replace(
        _step([4, 6], 0.4725),
        verification=VerificationResult.verified(0.0, passed=False),
        env_metrics={"exact_n10": 0.0},
    )
    n20_miss = replace(_step([5, 7, 8], 0.0), env_metrics={"exact_n20": 0.0})
    projection = WholeTrajectoryProjection(_config(), _Tokenizer())
    easy_batch = projection.project(
        [n10_hit, n10_miss],
        {"env_classes": ["cat_count", "cat_count"], "sampling_params": {"logprobs": True}},
    )
    hard_batch = projection.project(
        [n20_miss],
        {"env_classes": ["cat_count"], "sampling_params": {"logprobs": True}},
    )

    joined = concatenate_trajectory_batches([easy_batch, hard_batch], tis_lcs_alert_threshold=0.005)

    assert joined["rollout_metrics"]["environment/exact_n10"] == 0.5
    assert joined["rollout_metrics"]["environment/exact_n20"] == 0.0
    assert joined["rollout_metrics"]["generate/avg_tokens_non_zero_rewards"] == 1.0
    assert joined["rollout_metrics"]["generate/avg_tokens_zero_rewards"] == 2.5

    filtered = filter_trajectory_batch(joined, [0, 1])
    assert filtered["rollout_metrics"]["generate/avg_tokens_non_zero_rewards"] == 1.0
    assert filtered["rollout_metrics"]["generate/avg_tokens_zero_rewards"] == 2.0
    assert "environment/exact_n20" not in filtered["rollout_metrics"]

    unverified = projection.project(
        [
            replace(
                n20_miss,
                reward=RewardResult(unshaped_reward=0.0, optimization_reward=0.7),
                verification=VerificationResult.unavailable("judge unavailable"),
            )
        ],
        {"env_classes": ["cat_count"], "sampling_params": {"logprobs": True}},
    )
    assert unverified["rollout_metrics"]["generate/avg_tokens_non_zero_rewards"] == 0.0
    assert unverified["rollout_metrics"]["generate/avg_tokens_zero_rewards"] == 3.0
    unverified.pop("verification_results")
    mixed = concatenate_trajectory_batches([joined, unverified], tis_lcs_alert_threshold=0.005)
    assert mixed["rollout_metrics"]["generate/avg_tokens_non_zero_rewards"] == 2.0
    assert mixed["rollout_metrics"]["generate/avg_tokens_zero_rewards"] == 2.5
    unverified_only = filter_trajectory_batch(mixed, [3])
    assert unverified_only["rollout_metrics"]["generate/avg_tokens_non_zero_rewards"] == 3.0
    assert unverified_only["rollout_metrics"]["generate/avg_tokens_zero_rewards"] == 0.0
