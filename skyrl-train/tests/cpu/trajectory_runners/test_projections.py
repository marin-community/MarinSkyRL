from dataclasses import replace

from omegaconf import OmegaConf
from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult

from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.trajectory_runners.trajectory_processing import validate_trajectory_batch
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


def _step(
    response_ids, reward, *, captured_global_step=None, token_provenance="engine", first_token_policy_version=None
):
    outcome = float(sum(reward) if isinstance(reward, list) else reward)
    return AgentLoopOutput(
        evidence=RolloutEvidence(
            stop_reason="stop",
            prompt_token_ids=(1, 2),
            response_token_ids=tuple(response_ids),
            behavior_logprobs=tuple([-0.1] * len(response_ids)),
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
        captured_global_step=captured_global_step,
        first_token_policy_version=first_token_policy_version,
        token_provenance=token_provenance,
    )


def test_whole_trajectory_projection_preserves_one_sample_per_trajectory():
    projection = WholeTrajectoryProjection(_config(), _Tokenizer())
    output = projection.project(
        [_step([3, 4], [0.0, 1.0], captured_global_step=7)],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )

    assert output["response_ids"] == [[3, 4]]
    assert output["rewards"] == [[0.0, 1.0]]
    assert output["loss_masks"] == [[1, 1]]
    assert output["rollout_logprobs"] == [[-0.1, -0.1]]
    assert output["actual_global_step"] == 7
    assert output["rollout_metrics"]["generate/token_provenance/reconstructed_fraction"] == 0.0
    assert "trajectory_ids" not in output


def test_whole_trajectory_projection_adapts_masked_scalar_row_to_token_level_rewards():
    failed = _step([0], 0.0)
    failed = replace(
        failed,
        verification=VerificationResult.error(
            "SkyRL-Gym agent loop failed", diagnostics={"exception_type": "ConnectionError"}
        ),
        disposition=TrainingDisposition.mask("SkyRL-Gym agent loop failed", exception_type="ConnectionError"),
    )
    failed.error_treatment = "mask"

    projection = WholeTrajectoryProjection(_config(), _Tokenizer())
    output = projection.project(
        [_step([3, 4], [0.0, 1.0]), failed],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )

    assert output["rewards"] == [[0.0, 1.0], [0.0]]
    assert output["loss_masks"] == [[1, 1], [0]]
    assert output["exception_types"] == [None, "ConnectionError"]
    assert output["error_treatments"] == [None, "mask"]
    validate_trajectory_batch(2, output)


def test_step_wise_projection_preserves_group_identity_and_final_step():
    projection = StepWiseTrajectoryProjection(_config(), _Tokenizer())
    trajectory_id = TrajectoryID(instance_id="task", repetition_id=2)
    output = projection.project(
        [
            [
                _step([3], [1.0], captured_global_step=5),
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
    assert output["rollout_logprobs"] == [[-0.1], [-0.1, -0.1]]
    assert [(item.instance_id, item.repetition_id, item.step) for item in output["trajectory_ids"]] == [
        ("task", 2, 0),
        ("task", 2, 1),
    ]
    assert output["is_last_step"] == [False, True]
    assert output["actual_global_step"] == 5
    assert output["rollout_metrics"]["generate/token_provenance/reconstructed_fraction"] == 0.5


def test_step_wise_projection_preserves_student_topk_candidates():
    projection = StepWiseTrajectoryProjection(_config(), _Tokenizer())
    first = _step([3], 1.0)
    first.evidence = replace(
        first.evidence,
        student_topk_indices=((3, 4),),
        behavior_topk_logprobs=((-0.1, -1.1),),
    )
    second = _step([5], 0.0)
    second.evidence = replace(
        second.evidence,
        student_topk_indices=((5, 6),),
        behavior_topk_logprobs=((-0.2, -1.2),),
    )

    output = projection.project(
        [[first, second]],
        {"env_classes": ["math"], "trajectory_ids": [TrajectoryID("task", 0)], "sampling_params": {"logprobs": 2}},
    )

    assert output["student_topk_indices"] == [[[3, 4]], [[5, 6]]]
    assert output["behavior_topk_logprobs"] == [[[-0.1, -1.1]], [[-0.2, -1.2]]]


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


def test_whole_trajectory_projection_takes_the_oldest_first_token_version_of_the_group():
    projection = WholeTrajectoryProjection(_config(), _Tokenizer())
    output = projection.project(
        [
            _step([3], [1.0], first_token_policy_version=5),
            _step([4], [0.0], first_token_policy_version=None),
            _step([5], [0.0], first_token_policy_version=2),
        ],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )

    assert output["first_token_policy_version"] == 2
