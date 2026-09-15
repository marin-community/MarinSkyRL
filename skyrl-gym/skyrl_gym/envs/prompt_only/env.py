from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class PromptOnlyEnv(BaseTextEnv):
    """Terminal zero-reward environment for objectives supplied outside the verifier."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        del env_config, extras

    def step(self, action: str) -> BaseTextEnvStepOutput:
        del action
        return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={})
