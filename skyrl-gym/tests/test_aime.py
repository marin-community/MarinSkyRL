import pytest

from skyrl_gym.answer_tasks import grade_aime
from skyrl_gym.metrics import aggregate_for_task
from taskcompendium.grading import Outcome


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("Answer: \\boxed{42}", "42", 1.0),
        ("Answer: 42", "42", 1.0),
        ("Answer: \\boxed{43}", "42", -1.0),
        ("Answer: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", 1.0),
        ("Answer: \\boxed{0.5}", "\\frac{1}{2}", 1.0),
        ("Answer: \\boxed{15/56}", "\\frac{15}{56}", 1.0),
        ("Answer: \\boxed{7/26}", "\\dfrac{7}{26}", 1.0),
        ("Answer: \\boxed{109.2}", "\\frac{546}{5}", 1.0),
        ("Answer: \\boxed{25}", "025", 1.0),
        ("Answer: \\boxed{15/57}", "\\frac{15}{56}", -1.0),
        ("Answer: \\boxed{0.5}", "\\frac{1}{3}", -1.0),
        ("Answer: \\boxed{20:7}", "20:7", 1.0),
        ("Answer: \\boxed{40:14}", "20:7", 1.0),
        ("Answer: \\boxed{22:18}", "11:9", 1.0),
        ("Answer: \\boxed{7:20}", "20:7", -1.0),
        ("Answer: \\boxed{\\text{forty-two}}", "42", -1.0),
        # test EOS tokens
        ("<|im_start|>Answer: \\boxed{42}<|im_end|>", "42", 1.0),
        ("<|im_start|>Answer: \\boxed{42}im_end|>", "42", -1.0),
        ("Answer: \\boxed{42}<|eot_id|>", "42", 1.0),
        ("Answer: \\boxed{42}|eot_id|>", "42", -1.0),
    ],
)
def test_task_grade_preserves_math_equivalence(model_turn, output, ground_truth, expected):
    result = grade_aime(model_turn(output), {}, {"reward_model": {"ground_truth": ground_truth}})
    assert result.reward == expected


def test_aime_verifier_reports_failed_response_over_evaluation_budget(model_turn):
    result = grade_aime(
        model_turn("Answer: \\boxed{43}", token_count=5),
        {"evaluation_token_budget": 4},
        {"reward_model": {"ground_truth": "42"}},
    )
    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, -1.0)
    assert result.grade.passed is False
    assert result.metrics["over_evaluation_budget"] is True
    assert result.reward == -1.0


def test_aime_explicit_boxed_protocol_scores_answer_without_minerva_prefix(model_turn):
    response = "The final answer is \\boxed{42}.<|im_end|><|endoftext|>"
    extras = {"reward_model": {"ground_truth": "42"}}
    assert grade_aime(model_turn(response), {}, extras).reward == -1.0
    result = grade_aime(model_turn(response), {"strict_box_verify": True}, extras)
    assert result.reward == 1.0
    assert result.grade.diagnostics["prediction"] == "42"


def test_aime_aggregates_evaluation_budget_diagnostics_by_outcome():
    metrics = aggregate_for_task(
        "aime",
        [
            {"acc": True, "over_evaluation_budget": False, "answered_within_evaluation_budget": True},
            {"acc": True, "over_evaluation_budget": True, "answered_within_evaluation_budget": False},
            {"acc": False, "over_evaluation_budget": True, "answered_within_evaluation_budget": False},
            {"acc": False, "over_evaluation_budget": True, "answered_within_evaluation_budget": False},
        ],
    )

    assert metrics["over_evaluation_budget_fraction"] == pytest.approx(0.75)
    assert metrics["correct_over_evaluation_budget_fraction"] == pytest.approx(0.5)
    assert metrics["incorrect_over_evaluation_budget_fraction"] == pytest.approx(1.0)
    assert metrics["answered_within_evaluation_budget_fraction"] == pytest.approx(0.25)


def test_aime_verifier_marks_missing_answer_unparseable(model_turn):
    result = grade_aime(
        model_turn("I could not solve this.", token_count=4),
        {"evaluation_token_budget": 8},
        {"reward_model": {"ground_truth": "42"}},
    )
    assert result.grade.diagnostics["parseable_answer"] is False
    assert result.grade.diagnostics["answered_within_evaluation_budget"] is False


def test_aime_reward_policy_uses_generation_budget_from_evidence(model_turn):
    result = grade_aime(
        model_turn("Answer: \\boxed{42}", token_count=6, metadata={"generation_token_budget": 6}),
        {"length_penalty_weight": 0.5, "target_length": 2, "min_response_length": 0},
        {"reward_model": {"ground_truth": "42"}},
    )
    assert result.grade.reward == 1.0
    assert result.reward == pytest.approx(0.5)
    assert result.reward_components["length"] == pytest.approx(-0.5)
