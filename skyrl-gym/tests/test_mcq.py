import pytest

from skyrl_gym import get_data_contract
from skyrl_gym.answer_tasks import grade_mcq


@pytest.mark.parametrize("letter", ["D", "F", "H", "J", "Z"])
def test_task_rewards_boxed_answers_beyond_four_choices(model_turn, letter):
    result = grade_mcq(
        model_turn(f"The answer is ($\\boxed{{{letter}}}$)"), {}, {"reward_model": {"ground_truth": letter}}
    )
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "response,expected_reward",
    [
        (r"\boxed{A} revised to \boxed{B}", 1.0),
        (r"\boxed{B} revised to \boxed{A}", 0.0),
        (r"\boxed{a}", 1.0),
        (r"\boxed{AA}", 0.0),
        (r"\boxed{A", 0.0),
        ("Answer: A", 0.0),
    ],
)
def test_task_preserves_first_box_extraction(model_turn, response, expected_reward):
    result = grade_mcq(model_turn(response), {}, {"reward_model": {"ground_truth": "A"}})
    assert result.reward == expected_reward


def test_task_grade_agrees_with_dataset_preflight_for_a_lowercase_option(model_turn):
    contract = get_data_contract("mcq")
    response = r"reasoning... \boxed{f}"
    ground_truth = contract.validate_example("F", response, r"\boxed{A}")
    result = grade_mcq(model_turn(response), {}, {"reward_model": {"ground_truth": ground_truth}})
    assert result.grade.reward == 1.0
