import skyrl_gym
import pytest
from omegaconf import DictConfig
from skyrl_gym.verification import RolloutEvidence


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
    "output, ground_truth, stop_reason, expected",
    [
        ("Work.\n#### 42", "42", "stop", 1.0),
        ("Work.\n#### 42.0\n", "42", "eos", 1.0),
        ("Work.\n#### 1,234", "1234", "end_turn", 1.0),
        ("#### 42\nCorrection.\n#### 43", "42", "stop", 0.0),
        ("#### 43\nCorrection.\n#### 42", "42", "stop", 1.0),
        ("Work.\n#### 42\nMore text.", "42", "stop", 0.0),
        ("Work.\n#### 42", "42", "length", 0.0),
        ("Work.\n#### 42", "42", "error", 0.0),
        ("Work.\n#### 1,,234", "1234", "stop", 0.0),
        ("The answer is 42.", "42", "stop", 0.0),
    ],
)
def test_completed_final_line_reward(output, ground_truth, stop_reason, expected):
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"reward_method": "final_line"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": ground_truth}},
    )
    env.set_rollout_evidence(RolloutEvidence(response=output, stop_reason=stop_reason))
    assert env.step(output)["reward"] == expected
