"""Single-turn environment backed by Reasoning Gym task verifiers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.reasoning_gym.scoring import normalize_ground_truth, score_response

logger = logging.getLogger(__name__)


class ReasoningGymEnv(BaseTextEnv):
    """Score one generated task with its task-native verifier."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        reward_model = (extras or {}).get("reward_model")
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
        try:
            self.ground_truth = normalize_ground_truth(ground_truth)
        except (TypeError, ValueError):
            logger.exception("reasoning_gym: invalid reward_model.ground_truth=%r; scoring 0.", ground_truth)
            self.ground_truth = None

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.ground_truth is None:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"verifier_error": "invalid reward_model.ground_truth"},
            )
        reward = score_response(action, self.ground_truth)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})
