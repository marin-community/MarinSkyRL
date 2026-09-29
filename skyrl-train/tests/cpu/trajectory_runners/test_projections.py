from dataclasses import replace
import base64
import io
import pickle

import numpy as np

from omegaconf import OmegaConf
from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult

from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.trajectory_runners.trajectory_processing import validate_trajectory_batch
from skyrl_train.trajectory_runners.types import AgentLoopOutput, TrajectoryID
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors


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
    assert output["rollout_logprobs"] == [[-0.1, -0.1]]
    assert output["rollout_metrics"]["generate/token_provenance/reconstructed_fraction"] == 0.0
    assert "trajectory_ids" not in output


def test_whole_trajectory_projection_preserves_routes_and_fills_missing_rows():
    routed = _step([3, 4], [0.0, 1.0])
    routed.evidence = replace(routed.evidence, routed_experts=np.asarray([[[1, 2]], [[3, 4]]], dtype=np.int16))

    output = WholeTrajectoryProjection(_config(), _Tokenizer()).project(
        [routed, _step([5], [0.0])],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )

    np.testing.assert_array_equal(output["rollout_routed_experts"][0], [[[1, 2]], [[3, 4]]])
    np.testing.assert_array_equal(output["rollout_routed_experts"][1], [[[0, 0]]])


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
    assert output["rollout_logprobs"] == [[-0.1], [-0.1, -0.1]]
    assert [(item.instance_id, item.repetition_id, item.step) for item in output["trajectory_ids"]] == [
        ("task", 2, 0),
        ("task", 2, 1),
    ]
    assert output["is_last_step"] == [False, True]
    assert output["rollout_metrics"]["generate/token_provenance/reconstructed_fraction"] == 0.5


def test_step_wise_projection_preserves_routes():
    step = _step([3, 4], [0.0, 1.0])
    step.evidence = replace(step.evidence, routed_experts=np.asarray([[[1, 2]], [[3, 4]]], dtype=np.int16))

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


def test_encoded_routes_remain_compact_through_projection_and_training_collation():
    response_ids = list(range(1024))
    wire_rows = (np.arange(1026 * 4 * 2).reshape(1026, 4, 2) % 512).astype(np.uint16)
    stream = io.BytesIO()
    np.save(stream, wire_rows, allow_pickle=False)
    routes = normalize_routed_experts(base64.b64encode(stream.getvalue()).decode(), [10, 11, 12], response_ids)
    rollout = _step(response_ids, [0.0] * 1023 + [1.0])
    rollout.evidence = replace(rollout.evidence, routed_experts=routes)
    output = WholeTrajectoryProjection(_config(), _Tokenizer()).project(
        [rollout, _step([7], [0.0])],
        {"env_classes": None, "sampling_params": {"logprobs": True}},
    )
    projected = output["rollout_routed_experts"]
    # Nested Python containers inflate each token/layer into a tracked object.
    # The wire-to-trainer carrier must serialize at dense-array size instead.
    restored = pickle.loads(pickle.dumps(projected, protocol=5))
    assert len(pickle.dumps(projected, protocol=5)) < routes.nbytes + 2048
    np.testing.assert_array_equal(restored[0][:-1], wire_rows[3:])
    np.testing.assert_array_equal(restored[0][-1], np.zeros((4, 2)))
    np.testing.assert_array_equal(restored[1], np.zeros((1, 4, 2)))
    tokenizer = _Tokenizer()
    tokenizer.pad_token_id = 0
    packed = convert_prompts_responses_to_batch_tensors(
        tokenizer,
        [[10, 11, 12], [10]],
        [response_ids, [7]],
        output["rewards"],
        output["loss_masks"],
        routed_experts=restored,
        num_experts=512,
    )[6]
    np.testing.assert_array_equal(packed[0].numpy(), routes)
    np.testing.assert_array_equal(packed[1].numpy(), np.zeros((1024, 4, 2)))
