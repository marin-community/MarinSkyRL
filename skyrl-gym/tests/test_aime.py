import skyrl_gym
import pytest
from omegaconf import DictConfig
from skyrl_gym.envs.aime.env import AIMEEnv
from skyrl_gym.verification import RewardResult, RolloutEvidence, VerificationStatus


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("Answer: \\boxed{42}", "42", 1.0),
        ("Answer: 42", "42", 1.0),
        ("Answer: \\boxed{43}", "42", -1.0),
        ("Answer: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", 1.0),
        ("Answer: \\boxed{0.5}", "\\frac{1}{2}", -1.0),
        ("Answer: \\boxed{\\text{forty-two}}", "42", -1.0),
        # test EOS tokens
        ("<|im_start|>Answer: \\boxed{42}<|im_end|>", "42", 1.0),
        ("<|im_start|>Answer: \\boxed{42}im_end|>", "42", -1.0),
        ("Answer: \\boxed{42}<|eot_id|>", "42", 1.0),
        ("Answer: \\boxed{42}|eot_id|>", "42", -1.0),
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "aime",
        env_config=DictConfig({"env_class": "aime"}),
        extras={"reward_model": {"method": "rule", "ground_truth": ground_truth}},
    )
    step_output = env.step(output)
    assert step_output["reward"] == expected


def test_aime_verifier_reports_failed_response_over_evaluation_budget():
    env = skyrl_gym.make(
        "aime",
        env_config=DictConfig({"evaluation_token_budget": 4}),
        extras={"reward_model": {"ground_truth": "42"}},
    )
    env.set_rollout_evidence(
        RolloutEvidence(
            response="Answer: \\boxed{43}",
            stop_reason="stop",
            generated_token_count=5,
            response_token_ids=(1, 2, 3, 4, 5),
        )
    )

    step_output = env.step("Answer: \\boxed{43}")

    verification = step_output["verification"]
    assert verification.status is VerificationStatus.VERIFIED
    assert verification.score == -1.0
    assert verification.passed is False
    assert verification.diagnostics["over_evaluation_budget"] is True
    assert step_output["reward_result"] == RewardResult(unshaped_reward=-1.0, optimization_reward=-1.0)


def test_aime_aggregates_evaluation_budget_diagnostics_by_outcome():
    metrics = AIMEEnv.aggregate_metrics(
        [
            {"acc": True, "over_evaluation_budget": False, "answered_within_evaluation_budget": True},
            {"acc": True, "over_evaluation_budget": True, "answered_within_evaluation_budget": False},
            {"acc": False, "over_evaluation_budget": True, "answered_within_evaluation_budget": False},
            {"acc": False, "over_evaluation_budget": True, "answered_within_evaluation_budget": False},
        ]
    )

    assert metrics["over_evaluation_budget_fraction"] == pytest.approx(0.75)
    assert metrics["correct_over_evaluation_budget_fraction"] == pytest.approx(0.5)
    assert metrics["incorrect_over_evaluation_budget_fraction"] == pytest.approx(1.0)
    assert metrics["answered_within_evaluation_budget_fraction"] == pytest.approx(0.25)


def test_aime_aggregation_rejects_missing_budget_diagnostics():
    with pytest.raises(KeyError):
        AIMEEnv.aggregate_metrics([{"acc": True}])


def test_aime_verifier_marks_missing_answer_unparseable():
    env = skyrl_gym.make(
        "aime",
        env_config=DictConfig({"evaluation_token_budget": 8}),
        extras={"reward_model": {"ground_truth": "42"}},
    )
    env.set_rollout_evidence(
        RolloutEvidence(response="I could not solve this.", generated_token_count=4, response_token_ids=(1, 2, 3, 4))
    )

    verification = env.step("I could not solve this.")["verification"]

    assert verification.diagnostics["parseable_answer"] is False
    assert verification.diagnostics["answered_within_evaluation_budget"] is False


def test_aime_reward_policy_uses_generation_budget_from_evidence():
    env = skyrl_gym.make(
        "aime",
        env_config=DictConfig({"length_penalty_weight": 0.5, "target_length": 2, "min_response_length": 0}),
        extras={"reward_model": {"ground_truth": "42"}},
    )
    env.set_rollout_evidence(
        RolloutEvidence(
            response="Answer: \\boxed{42}",
            generated_token_count=6,
            response_token_ids=(1, 2, 3, 4, 5, 6),
            metadata={"generation_token_budget": 6},
        )
    )

    step_output = env.step("Answer: \\boxed{42}")

    assert step_output["verification"].score == 1.0
    assert step_output["reward"] == pytest.approx(0.5)
    assert step_output["reward_result"].components["length"] == pytest.approx(-0.5)
