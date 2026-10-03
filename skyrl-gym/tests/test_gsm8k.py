import skyrl_gym
import pytest
from omegaconf import DictConfig
from skyrl_gym.verification import RolloutEvidence


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("The answer is #### 42", "42", 1.0),
        ("The answer is #### 42", "43", 0.0),
        ("The answer is #### $10", "10", 1.0),
        ("The answer is #### $5", "5", 1.0),
        ("The answer is #### $10", "11", 0.0),
        ("The answer is #### $1,234", "1234", 1.0),
        # answer is not in the expected format
        ("The answer is 42", "42", 0.0),
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k", "reward_method": "strict"}),
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


def test_prepared_reward_model_extras_reach_the_verifier():
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k", "reward_method": "strict"}),
        extras={"reward_model": {"method": "rule", "ground_truth": "42"}},
    )
    assert env.ground_truth == "42"
    assert env.step("The answer is #### 42")["reward"] == 1.0


@pytest.mark.parametrize("method", ["strict", "flexible", "final_line"])
def test_invalid_reference_cannot_receive_format_credit(method):
    from skyrl_gym.envs.gsm8k.utils import compute_score

    assert compute_score("#### 10", None, method=method, format_score=0.5) == 0


@pytest.mark.parametrize("output", ["Draft #### 5\nCheck.\n#### 42", "Work.\n#### 42.0"])
def test_default_reward_grades_the_completed_final_line(output):
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({}),
        extras={"reward_spec": {"method": "rule", "ground_truth": "42"}},
    )
    env.set_rollout_evidence(RolloutEvidence(response=output, stop_reason="stop"))
    assert env.step(output)["reward"] == 1.0
