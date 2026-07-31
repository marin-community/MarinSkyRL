import json

import pytest

from skyrl_gym import get_data_contract


def test_aime_contract_normalizes_ground_truth_and_distinguishes_responses():
    contract = get_data_contract("aime")

    ground_truth = contract.validate_example(r"\boxed{42}", r"Answer: \boxed{42}", "")

    assert ground_truth == "42"


def test_ifeval_contract_canonicalizes_and_validates_constraint_specs():
    contract = get_data_contract("ifeval")

    ground_truth = contract.validate_example(
        {"keyword_list": ["alpha"], "func_name": "verify_keywords"},
        "contains alpha",
        "contains nothing",
    )

    assert json.loads(ground_truth) == {"func_name": "verify_keywords", "keyword_list": ["alpha"]}


@pytest.mark.parametrize(
    "env_id, ground_truth",
    [
        ("unknown", "42"),
        ("ifeval", {"func_name": "unknown_constraint"}),
        ("ifeval", {"func_name": "verify_keywords"}),
    ],
)
def test_contract_preflight_rejects_unknown_or_incomplete_verifier_specs(env_id, ground_truth):
    with pytest.raises(ValueError):
        get_data_contract(env_id).normalize_ground_truth(ground_truth)
