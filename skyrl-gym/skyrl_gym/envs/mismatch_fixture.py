"""Deterministic binary reward for small probe integration runs."""

import hashlib
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class MismatchFixtureEnv(BaseTextEnv):
    """Assign a stable binary reward from the generated response bytes."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        del env_config, extras

    def step(self, action: str) -> BaseTextEnvStepOutput:
        digest = hashlib.sha256(action.encode("utf-8")).digest()
        reward = float(digest[0] & 1)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})
