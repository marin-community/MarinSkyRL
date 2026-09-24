import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("The answer is #### 42", "42", 1.0),
        ("The answer is #### 42", "43", 0.0),
        # answer is not in the expected format
        ("The answer is 42", "42", 0.0),
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": ground_truth}},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(output)
    assert step_output["reward"] == expected


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("Reasoning.\n#### 42", "42", 1.0),
        ("Reasoning.\n#### 4,200  \n", "4200", 1.0),
        ("#### -1,234.5", "-1234.5", 1.0),
        ("#### 42\nMore text", "42", 0.0),
        ("The answer is #### 42", "42", 0.0),
        ("Reasoning.\n#### 43", "42", 0.0),
        ("#### 4,2", "42", 0.0),
        ("#### 42,", "42", 0.0),
        ("#### 4,,200", "4200", 0.0),
        ("#### 4..2", "4.2", 0.0),
    ],
)
def test_final_line_reward(output, ground_truth, expected):
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"reward_method": "strict_final_line"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": ground_truth}},
    )
    assert env.step(output)["reward"] == expected
