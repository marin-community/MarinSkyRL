from copy import deepcopy

import pytest

from skyrl_train.trajectory_runners.base import TrajectoryRequestBatch, TrajectoryRunner, TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_reward_shaping import (
    parse_trajectory_reward_shaping_config,
    shape_trajectory_rewards,
)
from skyrl_train.trajectory_runners.trajectory_processing import (
    concatenate_trajectory_batches,
    get_metrics_from_trajectory_batch,
)


def _output(
    responses: list[list[int]],
    rewards: list[float] | list[list[float]],
    stop_reasons: list[str],
    loss_masks: list[list[int]] | None = None,
) -> TrajectoryBatch:
    return {
        "prompt_token_ids": [[1] for _ in responses],
        "response_ids": responses,
        "rewards": rewards,
        "loss_masks": loss_masks or [[1] * len(response) for response in responses],
        "stop_reasons": stop_reasons,
        "rollout_metrics": {},
        "rollout_logprobs": None,
    }


def _loop_config(*, advantage_penalty_per_token: float = 0.1, max_advantage_penalty: float = 100.0):
    return {
        "enabled": True,
        "loop": {
            "max_period_tokens": 64,
            "tail_tokens": 256,
            "minimum_occurrences": 4,
            "advantage_penalty_per_token": advantage_penalty_per_token,
            "max_advantage_penalty": max_advantage_penalty,
        },
    }


@pytest.mark.parametrize(
    ("response", "expected_span"),
    [
        (list(range(12)) * 6, {"start": 36, "end": 72}),
        ([7] * 40, {"start": 3, "end": 40}),
        (list(range(6)) * 8, {"start": 18, "end": 48}),
    ],
)
def test_tail_loop_detection_returns_the_excess_repetition_span(response, expected_span):
    output = _output([response], [0.0], ["length"])

    shape_trajectory_rewards(output, _loop_config())

    assert output["reward_shaping_loop_spans"] == [[expected_span]]
    assert output["loop_advantages"][0][: expected_span["start"]] == [0.0] * expected_span["start"]
    assert output["loop_advantages"][0][expected_span["start"] :] == pytest.approx(
        [-0.1] * (expected_span["end"] - expected_span["start"])
    )


def test_tail_loop_detection_requires_minimum_occurrences():
    response = list(range(12)) * 3
    output = _output([response], [0.0], ["length"])

    shape_trajectory_rewards(output, _loop_config())

    assert output["reward_shaping_loop_spans"] == [[]]
    assert output["loop_advantages"] == [[0.0] * len(response)]


def test_passthrough_penalty_preserves_verified_outcome_and_normal_completion():
    output = _output(
        responses=[[1, 2], [3, 4]],
        rewards=[1.0, 1.0],
        stop_reasons=["turn_cap_exhausted", "complete"],
    )
    output["unshaped_rewards"] = [1.0, 1.0]
    output["error_treatments"] = ["passthrough", None]
    output["exception_types"] = ["TurnCapExhaustedError", None]

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "passthrough": {"penalty": 0.25},
        },
    )

    assert output["unshaped_rewards"] == [1.0, 1.0]
    assert output["rewards"] == pytest.approx([0.75, 1.0])
    assert [component["passthrough"] for component in output["reward_shaping_components"]] == [-0.25, 0.0]
    assert output["rollout_metrics"]["generate/reward_shaping/passthrough_incidence"] == 0.5


def test_loop_advantage_cap_scales_the_charged_span_without_touching_outcome_reward():
    response = list(range(4)) * 8
    output = _output([response], [1.0], ["length"])

    shape_trajectory_rewards(
        output,
        _loop_config(advantage_penalty_per_token=0.1, max_advantage_penalty=0.5),
    )

    charged_tokens = len(response) - 3 * 4
    realized_charge = 0.5 / charged_tokens
    assert output["rewards"] == [1.0]
    assert output["loop_advantages"] == [pytest.approx([0.0] * 12 + [-realized_charge] * charged_tokens)]
    assert sum(output["loop_advantages"][0]) == pytest.approx(-0.5)
    metrics = output["rollout_metrics"]
    assert metrics["generate/reward_shaping/loop_charged_tokens_mean"] == charged_tokens
    assert metrics["generate/reward_shaping/loop_advantage_per_token_mean"] == pytest.approx(-realized_charge)
    assert metrics["generate/reward_shaping/loop_incidence_correct"] == 1.0


def test_loop_followed_by_a_recovery_is_not_detected():
    output = _output(
        responses=[[1, 2] * 5 + [90, 91, 8, 9, 10]],
        rewards=[0.0],
        stop_reasons=["stop"],
        loss_masks=[[1] * 10 + [0, 0, 1, 1, 1]],
    )

    shape_trajectory_rewards(output, _loop_config())

    assert output["reward_shaping_loop_spans"] == [[]]
    assert output["loop_advantages"] == [[0.0] * 15]


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
        "passthrough": 0.0,
        "non_termination": 0.0,
        "overlong": 0.0,
        "successful_length": 0.0,
    }
    assert output["reward_shaping_components"][1]["successful_length"] == pytest.approx(-0.3)
    assert output["reward_shaping_components"][2] == {
        "passthrough": 0.0,
        "non_termination": 0.0,
        "overlong": 0.0,
        "successful_length": 0.0,
    }

    mean_reward, pass_at_two = get_metrics_from_trajectory_batch(output, ["short", "short", "failure"])
    assert mean_reward == pytest.approx((1.0 + 0.7) / 3)
    assert pass_at_two == pytest.approx(0.5)


def test_soft_overlong_penalty_applies_to_successful_and_failed_responses():
    output = _output(
        responses=[
            [1, 2],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4, 5, 6, 7],
        ],
        rewards=[1.0, 1.0, 0.0, 0.0],
        stop_reasons=["stop", "stop", "stop", "length"],
    )

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "overlong": {"l_max": 6, "l_cache": 4},
        },
    )

    assert output["unshaped_rewards"] == [1.0, 1.0, 0.0, 0.0]
    assert output["rewards"] == pytest.approx([1.0, 0.5, -0.5, -1.0])
    assert [component["overlong"] for component in output["reward_shaping_components"]] == pytest.approx(
        [0.0, -0.5, -0.5, -1.0]
    )
    assert all(component["successful_length"] == 0.0 for component in output["reward_shaping_components"])
    assert output["rollout_metrics"]["generate/reward_shaping/overlong_penalty_mean"] == pytest.approx(-0.5)
    assert output["rollout_metrics"]["generate/reward_shaping/overlong_incidence"] == pytest.approx(0.75)


@pytest.mark.parametrize(("penalty_scale", "expected_reward"), [(1.0, 0.0), (0.3, 0.7)])
def test_soft_overlong_penalty_scale_matches_reward_range(penalty_scale, expected_reward):
    output = _output([[1, 2, 3, 4]], [1.0], ["length"])

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "overlong": {"l_max": 4, "l_cache": 2, "penalty_scale": penalty_scale},
        },
    )

    assert output["rewards"] == pytest.approx([expected_reward])
    assert output["reward_shaping_components"][0]["overlong"] == pytest.approx(-penalty_scale)


def test_step_wise_soft_overlong_penalty_uses_full_trajectory_length():
    output = _output(
        responses=[[1, 2, 3, 4], [5, 6, 7, 8]],
        rewards=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        stop_reasons=["stop", "complete"],
    )
    output["is_last_step"] = [False, True]

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "overlong": {"l_max": 8, "l_cache": 2},
        },
    )

    # Neither four-token turn reaches the six-token penalty window. The complete
    # eight-token trajectory reaches l_max and receives the full penalty.
    assert output["rewards"] == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, pytest.approx(0.0)]]
    assert [component["overlong"] for component in output["reward_shaping_components"]] == pytest.approx([0.0, -1.0])
    assert output["rollout_metrics"]["generate/reward_shaping/overlong_penalty_mean"] == pytest.approx(-1.0)
    assert output["rollout_metrics"]["generate/reward_shaping/overlong_incidence"] == 1.0


def test_zero_cache_disables_soft_overlong_penalty():
    output = _output([[1, 2, 3, 4, 5, 6, 7, 8]], [1.0], ["length"])

    shape_trajectory_rewards(output, {"enabled": True, "overlong": {"l_max": 8, "l_cache": 0}})

    assert output["rewards"] == [1.0]
    assert output["reward_shaping_components"][0]["overlong"] == 0.0


def test_loop_advantage_stays_separate_from_scalar_trajectory_penalties():
    output = _output(
        responses=[[7, 8] * 5],
        rewards=[1.0],
        stop_reasons=["length"],
    )

    shape_trajectory_rewards(
        output,
        {
            "enabled": True,
            "loop": {
                "max_period_tokens": 2,
                "tail_tokens": 8,
                "minimum_occurrences": 4,
                "advantage_penalty_per_token": 0.1,
                "max_advantage_penalty": 0.2,
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

    assert output["rewards"] == pytest.approx([0.5])
    assert output["reward_shaping_components"] == [
        {"passthrough": 0.0, "non_termination": -0.3, "overlong": 0.0, "successful_length": -0.2}
    ]
    assert output["reward_shaping_loop_spans"] == [[{"start": 6, "end": 10}]]
    assert output["loop_advantages"] == [pytest.approx([0.0] * 6 + [-0.05] * 4)]
    assert output["reward_shaping_versions"] == [2]
    assert output["rollout_metrics"]["generate/reward_shaping/loop_incidence"] == 1.0
    assert output["rollout_metrics"]["generate/reward_shaping/non_termination_incidence"] == 1.0


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

    concatenated = concatenate_trajectory_batches([short, long], tis_lcs_alert_threshold=0.005)

    assert concatenated["rewards"] == pytest.approx([1.0, 0.7])
    assert concatenated["rollout_metrics"]["generate/reward_shaping/shaped_reward_mean"] == pytest.approx(0.85)
    assert concatenated["rollout_metrics"]["generate/reward_shaping/successful_length_penalty_mean"] == pytest.approx(
        -0.15
    )


def test_concatenation_preserves_later_passthrough_disposition():
    normal = _output([[1]], [1.0], ["complete"])
    passthrough = _output([[2]], [1.0], ["turn_cap_exhausted"])
    passthrough["exception_types"] = ["TurnCapExhaustedError"]
    passthrough["error_treatments"] = ["passthrough"]

    concatenated = concatenate_trajectory_batches([normal, passthrough], tis_lcs_alert_threshold=0.005)

    assert concatenated["exception_types"] == [None, "TurnCapExhaustedError"]
    assert concatenated["error_treatments"] == [None, "passthrough"]


class _SharedShapingRunner(TrajectoryRunner):
    trajectory_runner_cfg = {
        "trajectory_reward_shaping": {
            "enabled": True,
            "non_termination": {"penalty": 0.4},
        }
    }

    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        return _output([[1, 2]], [1.0], ["length"])


@pytest.mark.asyncio
async def test_trajectory_runner_applies_shared_shaping_after_generation():
    output = await _SharedShapingRunner().run({})

    assert output["unshaped_rewards"] == [1.0]
    assert output["rewards"] == pytest.approx([0.6])


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"schema_version": 3}, "schema_version"),
        ({"loop": {"max_period_tokens": 0}}, "loop.max_period_tokens"),
        ({"loop": {"max_period_tokens": 4, "tail_tokens": 15}}, "loop.tail_tokens"),
        ({"loop": {"minimum_occurrences": 1}}, "loop.minimum_occurrences"),
        ({"loop": {"penalty_per_occurrence": 0.1}}, "unknown loop settings"),
        (
            {"loop": {"advantage_penalty_per_token": 0.1, "max_advantage_penalty": 0.0}},
            "loop.max_advantage_penalty must be positive",
        ),
        ({"passthrough": {"penalty": -0.1}}, "passthrough.penalty"),
        ({"non_termination": {"penalty": -0.1}}, "non_termination.penalty"),
        ({"non_termination": {"accepted_stop_reasons": "stop"}}, "accepted_stop_reasons"),
        ({"successful_length": {"free_tokens": -1}}, "successful_length.free_tokens"),
        ({"overlong": {"l_max": 0, "l_cache": -1}}, "overlong.l_cache must be non-negative"),
        ({"overlong": {"l_max": 8, "l_cache": 9}}, "overlong.l_cache must not exceed overlong.l_max"),
        ({"overlong": {"penalty_scale": -0.1}}, "overlong.penalty_scale must be non-negative"),
    ],
)
def test_invalid_shaping_config_fails_before_generation(config, message):
    with pytest.raises(ValueError, match=message):
        parse_trajectory_reward_shaping_config(config)
