from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.mcq.utils import extract_mcq_answer


class MCQEnv(BaseTextEnv):
    """Single-turn multiple-choice verifier.

    ``ground_truth`` is the correct option letter (e.g. ``"A"``, or ``"H"`` in a
    ten-choice prompt). The reward is 1 when the boxed answer in the response
    matches the expected letter, 0 otherwise.
    """

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        assert "reward_model" in extras, "reward_model field is required"
        assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
        self.ground_truth = str(extras["reward_model"]["ground_truth"]).strip().upper()

    def step(self, action: str) -> BaseTextEnvStepOutput:
        answer = extract_mcq_answer(action)
        reward = 1.0 if answer is not None and answer == self.ground_truth else 0.0
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})
