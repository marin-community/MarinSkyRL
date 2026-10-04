import pytest
from omegaconf import OmegaConf

from skyrl_gym import get_data_contract
from skyrl_gym.envs.mcq.env import MCQEnv


def _env(ground_truth: str) -> MCQEnv:
    return MCQEnv(OmegaConf.create(), extras={"reward_model": {"ground_truth": ground_truth}})


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
def test_env_preserves_first_box_extraction(response, expected_reward):
    assert _env("A").step(response)["reward"] == expected_reward


def test_env_rewards_last_alphabet_option():
    assert _env("Z").step(r"\boxed{Z}")["reward"] == 1.0
