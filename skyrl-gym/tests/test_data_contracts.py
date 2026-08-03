import json

import pytest

from skyrl_gym import get_data_contract


LCB_CORRECT_RESPONSE = """```python
def main():
    value = input()
    print(value[::-1])

if __name__ == "__main__":
    main()
```"""
LCB_WRONG_RESPONSE = """```python
print(input())
```"""


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
    "source_ground_truth",
    [
        {"inputs": ["abc\n"], "outputs": ["cba\n"]},
        {
            "language": "python",
            "test_cases": [
                {
                    "fn_name": None,
                    "input": "abc\n",
                    "output": "cba\n",
                    "type": "stdin_stdout",
                }
            ],
        },
    ],
)
def test_lcb_contract_normalizes_source_schemas_and_executes_two_sided_preflight(source_ground_truth):
    contract = get_data_contract("lcb")

    ground_truth = contract.validate_example(
        source_ground_truth,
        LCB_CORRECT_RESPONSE,
        LCB_WRONG_RESPONSE,
    )

    assert json.loads(ground_truth) == [{"input": "abc\n", "output": "cba\n", "testtype": "stdin"}]


def test_lcb_contract_normalizes_and_executes_functional_cases():
    contract = get_data_contract("lcb")

    ground_truth = contract.validate_example(
        {"inputs": [[[1, 2, 3]]], "outputs": [6], "fn_name": "total"},
        "```python\ndef total(values):\n    return sum(values)\n```",
        "```python\ndef total(values):\n    return 0\n```",
    )

    assert json.loads(ground_truth) == [
        {
            "input": "[1, 2, 3]",
            "metadata": {"func_name": "total"},
            "output": "6",
            "testtype": "functional",
        }
    ]


@pytest.mark.parametrize(
    "ground_truth, error",
    [
        ({"language": "rust", "test_cases": []}, "only Python"),
        ({"inputs": ["1\n"], "outputs": []}, "equally sized"),
        (
            [{"input": "1\n", "output": "1\n", "testtype": "functional"}],
            "function name",
        ),
        (
            [
                {"input": "1\n", "output": "1\n", "testtype": "stdin"},
                {
                    "input": "1",
                    "output": "1",
                    "testtype": "functional",
                    "metadata": {"func_name": "identity"},
                },
            ],
            "cannot mix",
        ),
        (
            [
                {
                    "input": "1",
                    "output": "1",
                    "testtype": "functional",
                    "metadata": {"func_name": "first"},
                },
                {
                    "input": "1",
                    "output": "1",
                    "testtype": "functional",
                    "metadata": {"func_name": "second"},
                },
            ],
            "same function name",
        ),
    ],
)
def test_lcb_contract_rejects_unlaunchable_test_schemas(ground_truth, error):
    with pytest.raises((TypeError, ValueError), match=error):
        get_data_contract("lcb").normalize_ground_truth(ground_truth)


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
