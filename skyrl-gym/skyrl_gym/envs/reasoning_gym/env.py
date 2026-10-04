"""Single-turn environment backed by Reasoning Gym task verifiers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.reasoning_gym.scoring import normalize_ground_truth, score_response

logger = logging.getLogger(__name__)


class ReasoningGymEnv(BaseTextEnv):
    """Score one generated task with its task-native verifier."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        self.structured_chat = env_config.get("structured_chat", False)
        self.verifyit_enabled = bool(env_config.get("verifyit_enabled", False))
        reward_model = (extras or {}).get("reward_model")
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
        try:
            self.ground_truth = normalize_ground_truth(ground_truth)
        except (TypeError, ValueError):
            logger.exception("reasoning_gym: invalid reward_model.ground_truth=%r; scoring 0.", ground_truth)
            self.ground_truth = None

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        metadata = {"chat_completion_params": {}} if self.structured_chat else {}
        return prompt, metadata

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.ground_truth is None:
            if self.verifyit_enabled:
                from skyrl_gym.verification import VerificationResult

                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={},
                    verification=VerificationResult.error("invalid Reasoning Gym task"),
                )
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"verifier_error": "invalid reward_model.ground_truth"},
            )
        try:
            reward = score_response(action, self.ground_truth, verifyit_enabled=self.verifyit_enabled)
        except (RuntimeError, ValueError) as error:
            from skyrl_gym.verification import VerificationResult

            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={},
                verification=VerificationResult.error(
                    "verifier failed", diagnostics={"error_type": type(error).__name__}
                ),
            )
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})
