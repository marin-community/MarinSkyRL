from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class PreferenceEnv(BaseTextEnv):
    """Placeholder verifier for preference / RLHF sources.

    At runtime the reward comes from a reward model (not a deterministic
    checker).  This env simply records the chosen response so that dataset
    preparation can validate that chosen and rejected differ.
    """

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={})
