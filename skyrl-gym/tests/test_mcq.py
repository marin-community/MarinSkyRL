import json
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

from skyrl_gym import get_data_contract
from skyrl_gym.envs.mcq.env import MCQEnv


def _env(ground_truth: str, *, verifyit_enabled=False) -> MCQEnv:
    return MCQEnv(
        OmegaConf.create({"verifyit_enabled": verifyit_enabled}),
        extras={"reward_model": {"ground_truth": ground_truth}},
    )


@pytest.mark.parametrize("letter", ["D", "F", "H", "J"])
def test_env_rewards_boxed_answers_beyond_four_choices(letter):
    env = _env(letter)

    assert env.step(f"The answer is ($\\boxed{{{letter}}}$)")["reward"] == 1.0


def test_env_matches_lowercase_boxed_answers_and_rejects_wrong_letters():
    env = _env("H")

    assert env.step("The answer is ($\\boxed{h}$)")["reward"] == 1.0
    assert env.step("The answer is ($\\boxed{D}$)")["reward"] == 0.0
    assert env.step("The answer is \\boxed{HD}")["reward"] == 0.0
    assert env.step("No boxed answer here.")["reward"] == 0.0


def test_env_constructs_from_a_prepared_row_extras_mapping():
    extras = {
        "data_source": "nvidia/OpenScience",
        "reward_model": {"ground_truth": "H"},
        "extra_info": {"split": "train", "index": 4, "subset": "OS-Q2.5-32B-10"},
    }

    env = MCQEnv(OmegaConf.create(), extras=extras)

    assert env.step("The answer is ($\\boxed{H}$)")["reward"] == 1.0
    assert env.step("The answer is ($\\boxed{A}$)")["reward"] == 0.0


@pytest.mark.parametrize(
    ("ground_truth", "response"),
    [
        ("H", "The answer is ($\\boxed{H}$)"),
        ("F", "reasoning... \\boxed{f}"),
        ("D", "\\boxed{D}"),
        ("J", "\\boxed{B}"),
        ("A", "no boxed answer"),
    ],
)
def test_env_agrees_with_the_preparation_contract(ground_truth, response):
    contract = get_data_contract("mcq")

    expected = 1.0 if contract.is_correct(response, ground_truth) else 0.0

    assert _env(ground_truth).step(response)["reward"] == expected


@pytest.mark.parametrize(
    ("response", "expected_reward"),
    [
        (r"\boxed{A} revised to \boxed{B}", 1.0),
        (r"\boxed{B} revised to \boxed{A}", 0.0),
        (r"\boxed{a}", 1.0),
        (r"\boxed{AA}", 0.0),
        (r"\boxed{A", 0.0),
        ("Answer: A", 0.0),
    ],
)
@pytest.mark.parametrize("verifyit_enabled", [False, True])
def test_env_preserves_first_box_extraction(response, expected_reward, verifyit_enabled):
    assert _env("A", verifyit_enabled=verifyit_enabled).step(response)["reward"] == expected_reward


def test_env_rewards_last_alphabet_option():
    assert _env("Z").step(r"\boxed{Z}")["reward"] == 1.0


def test_default_verifier_grading_runs_without_verifyit():
    program = r"""
import importlib.abc
import json
import sys
class MissingVerifyit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "verifyit" or fullname.startswith("verifyit."):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, MissingVerifyit())
from omegaconf import OmegaConf
import skyrl_gym
results = []
for route, expected, correct, wrong in [
    ("mcq", "H", r"\boxed{h}", r"\boxed{A}"),
    ("aime", "42", r"Answer: \boxed{42}", r"Answer: \boxed{43}"),
    ("gsm8k", "42", "#### 42", "#### 43"),
]:
    env = skyrl_gym.make(route, env_config=OmegaConf.create(), extras={"reward_model": {"ground_truth": expected}})
    results.append([env.step(correct)["reward"], env.step(wrong)["reward"]])
print(json.dumps(results))
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [[1.0, 0.0], [1.0, -1.0], [1.0, 0.0]]


@pytest.mark.parametrize("reference", ["AB", "", None])
@pytest.mark.parametrize("response", [r"\boxed{A}", "No answer"])
def test_enabled_mcq_reports_invalid_reference_without_credit(reference, response):
    result = _env(reference, verifyit_enabled=True).step(response)
    assert result["reward"] == 0.0
    assert result["verification"].status.value == "error"
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"
