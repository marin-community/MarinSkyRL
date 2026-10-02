from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ground_truth_from_extras


class PreferenceEnv(BaseTextEnv):
    """Placeholder verifier for preference / RLHF sources.

    At runtime the reward comes from a reward model (not a deterministic
    checker).  This env simply records the chosen response so that dataset
    preparation can validate that chosen and rejected differ.
    """

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        self.ground_truth = ground_truth_from_extras(extras or {})

    def step(self, action: str) -> BaseTextEnvStepOutput:
        return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={})
