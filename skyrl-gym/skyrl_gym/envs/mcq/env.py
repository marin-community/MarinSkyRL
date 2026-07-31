import re
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class MCQEnv(BaseTextEnv):
    """Single-turn multiple-choice verifier.

    ``ground_truth`` is the correct option letter (e.g. ``"A"``).
    The reward is 1 when the boxed answer in the response matches the
    expected letter, 0 otherwise.
    """

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = str(extras["reward_spec"]["ground_truth"]).strip().upper()

    def step(self, action: str) -> BaseTextEnvStepOutput:
        reward = 1.0 if _extract_mcq_answer(action) == self.ground_truth else 0.0
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})


def _extract_mcq_answer(response: str) -> str | None:
    match = re.search(r"\\boxed\{([A-D])\}", response)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{([a-d])\}", response)
    if match:
        return match.group(1).upper()
    return None
