import json

import pytest
import skyrl_gym
from omegaconf import DictConfig

from skyrl_gym import get_data_contract
from skyrl_gym.envs.nupa.utils import digit_parts, extract_answer, full_answer
from skyrl_gym.verification import RolloutEvidence

INTEGER = "Integer"
FLOAT = "Float"
FRACTION = "Fraction"
SCIENTIFIC = "ScientificNotation"


def make_nupa_env(ground_truth):
    return skyrl_gym.make(
        "nupa",
        env_config=DictConfig({"env_class": "nupa"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": json.dumps(ground_truth)}},
    )


@pytest.mark.parametrize(
    ("prediction", "answer_format", "expected"),
    [
        ("The answer is 9.9, because 9.9 is larger.", FLOAT, "9.9"),
        ("I think the answer is 9.9", FLOAT, None),
        ("So the answer is 123", INTEGER, "123"),
        ("5.04E+04", SCIENTIFIC, "5.04e04"),
        (r"<|start_think|>Maybe \boxed{9}<|end_think|>\(\boxed{1/1511}\).", FRACTION, "1/1511"),
        ("<|start_think|>Maybe 9<|end_think|>42", INTEGER, "42"),
        (r"<|start_think|>\boxed{9}", INTEGER, None),
        (r"The answer is \boxed{9.90}.", FLOAT, "9.90"),
        ("Therefore, the final answer is 2.547762432164e18.", SCIENTIFIC, "2.547762432164e18"),
        (r"\boxed{1,234}", INTEGER, None),
        (r"\boxed{9.90}", INTEGER, None),
        ("Result: 9.9\nI need to check that", FLOAT, None),
    ],
)
def test_extract_answer_matches_the_nupa_loose_eval_protocol(prediction, answer_format, expected):
    assert extract_answer(prediction, answer_format) == expected


def test_digit_components_preserve_representation_sensitivity():
    assert digit_parts("9.9", FLOAT) == ("9", "9")
    assert digit_parts("1/2", FRACTION) == ("1", "2")
    assert digit_parts("5.04e+04", SCIENTIFIC) == ("5", "04", "04")
    assert digit_parts("1.2.3", FLOAT) == ("1", "23")
    assert digit_parts("123", FLOAT) == ("", "")
    assert full_answer("9.90", FLOAT) == "9.90"
    assert full_answer("1,234", INTEGER) is None


@pytest.mark.parametrize(
    ("response", "ground_truth", "reward", "digit_match"),
    [
        ("9.9", {"answer": "9.9", "answer_format": FLOAT}, 1.0, 1.0),
        ("The answer is 9.9", {"answer": "9.9", "answer_format": FLOAT}, 1.0, 1.0),
        ("9.90", {"answer": "9.9", "answer_format": FLOAT}, 0.0, 1.0),
        ("9.11", {"answer": "9.9", "answer_format": FLOAT}, 0.0, 0.5),
        ("one half", {"answer": "1/2", "answer_format": FRACTION}, 0.0, 0.0),
        ("123", {"answer": "123", "answer_format": INTEGER}, 1.0, 1.0),
        ("5.04e4", {"answer": "5.04e4", "answer_format": SCIENTIFIC}, 1.0, 1.0),
    ],
)
def test_env_rewards_the_policy_eval_exact_match_metric(response, ground_truth, reward, digit_match):
    step = make_nupa_env(ground_truth).step(response)

    assert step["reward"] == reward
    assert step["metadata"]["exact_match"] == reward
    assert step["metadata"]["digit_match"] == digit_match
    assert step["metadata"]["acc"] is (reward == 1.0)


def test_env_reports_no_answer_metrics_for_unparseable_responses():
    step = make_nupa_env({"answer": "1/2", "answer_format": FRACTION}).step("I cannot solve this.")

    assert step["reward"] == 0.0
    assert step["metadata"]["format_valid"] == 0.0
    assert step["metadata"]["no_answer"] == 1.0
    assert step["metadata"]["dlength"] == 2.0


def test_env_scores_zero_on_malformed_ground_truth_instead_of_crashing():
    for bad in ("not json", json.dumps({"answer": "1", "answer_format": "Roman"}), json.dumps({"answer": ""})):
        step = make_nupa_env(bad).step("123")

        assert step["reward"] == 0.0


def test_env_aggregates_metric_means():
    metrics = [
        make_nupa_env({"answer": "9.9", "answer_format": FLOAT}).step("9.9")["metadata"],
        make_nupa_env({"answer": "9.9", "answer_format": FLOAT}).step("9.11")["metadata"],
    ]

    aggregated = skyrl_gym.make("nupa", env_config=DictConfig({}), extras={}).aggregate_metrics(metrics)

    assert aggregated["nupa/acc"] == 0.5
    assert aggregated["nupa/exact_match"] == 0.5
    assert aggregated["nupa/digit_match"] == 0.75


def test_data_contract_validates_ground_truth_against_the_runtime_verifier():
    contract = get_data_contract("nupa")

    normalized = contract.validate_example(
        {"answer": "1/1511", "answer_format": FRACTION},
        "The simplified fraction is \\(\\boxed{1/1511}\\).",
        "The simplified fraction is \\(\\boxed{1/1512}\\).",
    )

    assert json.loads(normalized) == {"answer": "1/1511", "answer_format": FRACTION}


def test_data_contract_rejects_malformed_ground_truth():
    contract = get_data_contract("nupa")

    with pytest.raises(ValueError):
        contract.validate_example({"answer": "1"}, "1", "2")
    with pytest.raises(ValueError):
        contract.validate_example({"answer": "9.9", "answer_format": "Hexadecimal"}, "9.9", "1.0")


def test_verifier_uses_rollout_evidence_response():
    from skyrl_gym.envs.nupa.verifier import NUPAVerifier
    from skyrl_gym.verification import VerificationStatus

    verdict = NUPAVerifier(ground_truth=json.dumps({"answer": "42", "answer_format": INTEGER})).verify(
        RolloutEvidence(response="So the answer is: 42", stop_reason="stop")
    )

    assert verdict.status is VerificationStatus.VERIFIED
    assert verdict.passed is True
    assert verdict.diagnostics["prediction"] == "42"
