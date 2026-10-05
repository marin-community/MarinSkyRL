from omegaconf import OmegaConf
import pytest

from skyrl_gym.envs.gsm8k.multi_turn_env import GSM8kMultiTurnEnv
from skyrl_gym.verification import VerificationStatus


@pytest.mark.parametrize("enabled", [False, True])
def test_final_turn_preserves_format_reward_and_success(enabled):
    env = GSM8kMultiTurnEnv(
        OmegaConf.create({"verifyit_enabled": enabled}),
        extras={"reward_spec": {"ground_truth": "42"}, "max_turns": 2},
    )
    first = env.step("#### 41")
    assert first["reward"] == 0.1
    assert not first["done"]
    final = env.step("#### 42")
    assert final["reward"] == 1.0
    assert final["done"]
    assert final["observations"] == []


def test_invalid_task_terminates_with_minimum_score():
    env = GSM8kMultiTurnEnv(
        OmegaConf.create({"verifyit_enabled": True}),
        extras={"reward_spec": {"ground_truth": None}, "max_turns": 2},
    )
    result = env.step("#### 42")
    assert result["reward"] == 0.0
    assert result["done"]
    assert result["verification"].status == VerificationStatus.ERROR
    assert result["verification"].diagnostics["verifyit_status"] == "invalid_task"
