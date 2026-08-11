from copy import deepcopy

import pytest

from skyrl_train.generators.base import GeneratorInput, GeneratorInterface, GeneratorOutput
from skyrl_train.generators.trajectory_reward_shaping import (
    infer_stop_reason,
    parse_trajectory_reward_shaping_config,
    shape_trajectory_rewards,
)
from skyrl_train.generators.utils import concatenate_generator_outputs, get_metrics_from_generator_output


def _output(
    responses: list[list[int]],
    rewards: list[float] | list[list[float]],
    stop_reasons: list[str],
    loss_masks: list[list[int]] | None = None,
) -> GeneratorOutput:
    return {
        "prompt_token_ids": [[1] for _ in responses],
        "response_ids": responses,
        "rewards": rewards,
        "loss_masks": loss_masks or [[1] * len(response) for response in responses],
        "stop_reasons": stop_reasons,
        "rollout_metrics": {},
        "rollout_logprobs": None,
    }


def test_successful_length_penalty_only_changes_positive_outcomes():
    output = _output(
        responses=[[1, 2], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8]],
        rewards=[1.0, 1.0, 0.0],
        stop_reasons=["stop", "stop", "stop"],
    )

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "successful_length": {
                "free_tokens": 2,
                "penalty_per_token": 0.1,
                "max_penalty": 0.5,
            },
        },
    )

    assert output["unshaped_rewards"] == [1.0, 1.0, 0.0]
    assert output["rewards"] == pytest.approx([1.0, 0.7, 0.0])
    assert output["reward_shaping_components"][0] == {
        "loop": 0.0,
        "non_termination": 0.0,
        "successful_length": 0.0,
    }
    assert output["reward_shaping_components"][1]["successful_length"] == pytest.approx(-0.3)
    assert output["reward_shaping_components"][2] == {
        "loop": 0.0,
        "non_termination": 0.0,
        "successful_length": 0.0,
    }

    mean_reward, pass_at_two = get_metrics_from_generator_output(output, ["short", "short", "failure"])
    assert mean_reward == pytest.approx((1.0 + 0.7) / 3)
    assert pass_at_two == pytest.approx(0.5)


def test_loop_and_non_termination_penalties_compose_from_raw_trajectory():
    output = _output(
        responses=[[7, 8, 7, 8, 7, 8]],
        rewards=[1.0],
        stop_reasons=["length"],
    )

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "loop": {
                "window_tokens": 2,
                "minimum_occurrences": 2,
                "penalty_per_occurrence": 0.1,
                "max_penalty": 0.2,
            },
            "non_termination": {
                "penalty": 0.3,
                "accepted_stop_reasons": ["stop", "complete"],
            },
            "successful_length": {
                "free_tokens": 2,
                "penalty_per_token": 0.05,
                "max_penalty": 0.2,
            },
        },
    )

    assert output["rewards"] == pytest.approx([0.3])
    assert output["reward_shaping_components"] == [
        {"loop": -0.2, "non_termination": -0.3, "successful_length": -0.2}
    ]
    assert output["reward_shaping_loop_spans"] == [[{"start": 2, "end": 6}]]
    assert output["reward_shaping_versions"] == [1]
    assert output["rollout_metrics"]["generate/reward_shaping/loop_incidence"] == 1.0
    assert output["rollout_metrics"]["generate/reward_shaping/non_termination_incidence"] == 1.0


def test_loop_detection_does_not_cross_assistant_turn_boundaries():
    output = _output(
        responses=[[10, 11, 90, 91, 10, 11]],
        rewards=[0.0],
        stop_reasons=["complete"],
        loss_masks=[[1, 1, 0, 0, 1, 1]],
    )

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "loop": {
                "window_tokens": 2,
                "minimum_occurrences": 2,
                "penalty_per_occurrence": 0.1,
                "max_penalty": 0.5,
            },
        },
    )

    assert output["rewards"] == [0.0]
    assert output["reward_shaping_loop_spans"] == [[]]
    assert output["rollout_metrics"]["generate/reward_shaping/loop_incidence"] == 0.0


def test_token_rewards_receive_trajectory_penalty_on_last_trainable_token():
    output = _output(
        responses=[[1, 2, 3]],
        rewards=[[0.0, 0.0, 1.0]],
        stop_reasons=["length"],
        loss_masks=[[1, 1, 1]],
    )

    shape_trajectory_rewards(
        output,
        {"enabled": True, "non_termination": {"penalty": 0.25}},
    )

    assert output["unshaped_rewards"] == [1.0]
    assert output["rewards"] == [[0.0, 0.0, 0.75]]


def test_step_wise_shaping_accumulates_length_and_penalizes_final_step():
    output = _output(
        responses=[[1, 2], [3, 4, 5]],
        rewards=[[0.0, 0.5], [0.0, 0.0, 1.0]],
        stop_reasons=["stop", "complete"],
    )
    output["is_last_step"] = [False, True]

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "successful_length": {
                "free_tokens": 2,
                "penalty_per_token": 0.1,
                "max_penalty": 0.5,
            },
        },
    )

    assert output["rewards"] == [[0.0, 0.5], [0.0, 0.0, pytest.approx(0.7)]]
    assert output["reward_shaping_components"][0]["successful_length"] == 0.0
    assert output["reward_shaping_components"][1]["successful_length"] == pytest.approx(-0.3)
    assert output["rollout_metrics"]["generate/reward_shaping/response_tokens_mean"] == 5.0


def test_disabled_trajectory_shaping_preserves_output_exactly():
    output = _output([[1, 2, 3]], [1.0], ["length"])
    original = deepcopy(output)

    shape_trajectory_rewards(output, {"enabled": False})

    assert output == original


def test_concatenation_recomputes_shaping_metrics_from_retained_components():
    short = _output([[1, 2]], [1.0], ["stop"])
    long = _output([[1, 2, 3, 4, 5]], [1.0], ["stop"])
    config = {
        "enabled": True,
        "successful_length": {
            "free_tokens": 2,
            "penalty_per_token": 0.1,
            "max_penalty": 0.5,
        },
    }
    shape_trajectory_rewards(short, config)
    shape_trajectory_rewards(long, config)

    concatenated = concatenate_generator_outputs([short, long])

    assert concatenated["rewards"] == pytest.approx([1.0, 0.7])
    assert concatenated["rollout_metrics"]["generate/reward_shaping/shaped_reward_mean"] == pytest.approx(0.85)
    assert concatenated["rollout_metrics"]["generate/reward_shaping/successful_length_penalty_mean"] == pytest.approx(
        -0.15
    )


class _SharedShapingGenerator(GeneratorInterface):
    generator_cfg = {
        "trajectory_reward_shaping": {
            "enabled": True,
            "non_termination": {"penalty": 0.4},
        }
    }

    async def _generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        return _output([[1, 2]], [1.0], ["length"])


@pytest.mark.asyncio
async def test_generator_interface_applies_shared_shaping_after_generation():
    output = await _SharedShapingGenerator().generate({})

    assert output["unshaped_rewards"] == [1.0]
    assert output["rewards"] == pytest.approx([0.6])


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"loop": {"window_tokens": 0}}, "loop.window_tokens"),
        ({"loop": {"minimum_occurrences": 1}}, "loop.minimum_occurrences"),
        ({"non_termination": {"penalty": -0.1}}, "non_termination.penalty"),
        ({"non_termination": {"accepted_stop_reasons": "stop"}}, "accepted_stop_reasons"),
        ({"successful_length": {"free_tokens": -1}}, "successful_length.free_tokens"),
    ],
)
def test_invalid_shaping_config_fails_before_generation(config, message):
    with pytest.raises(ValueError, match=message):
        parse_trajectory_reward_shaping_config(config)


@pytest.mark.parametrize(
    ("response_ids", "expected"),
    [
        ([1, 2, 99], "stop"),
        ([1, 2, 3, 4], "length"),
        ([1, 2], "stop"),
    ],
)
def test_stop_reason_inference_distinguishes_eos_and_budget_exhaustion(response_ids, expected):
    assert infer_stop_reason(response_ids, eos_token_id=99, max_generate_length=4) == expected
