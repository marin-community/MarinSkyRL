import logging
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.mcq.utils import extract_mcq_answer
from skyrl_gym.verification import VerificationResult


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
        self.raw_ground_truth = extras["reward_model"]["ground_truth"]
        self.ground_truth = str(self.raw_ground_truth).strip().upper()
        self.verifyit_enabled = bool(env_config.get("verifyit_enabled", False))

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.verifyit_enabled:
            try:
                from skyrl_gym.envs.mcq.verifyit import MCQPolicy, grade_mcq
            except Exception as error:
                logging.getLogger(__name__).exception("MCQ verifier import failed")
                reward = 0.0
                details = {
                    "error_type": "verification_error",
                    "verifyit_status": "infra_error",
                    "cause_error_type": type(error).__name__,
                    "preparation_stage": "mcq_import",
                    "error_message": str(error),
                }
            else:
                reward, details = grade_mcq(
                    action, {"expected_answer": self.raw_ground_truth}, policy=MCQPolicy.FIRST_BOX
                )
            if details.get("error_type"):
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata=details,
                    verification=VerificationResult.error("MCQ verification failed", diagnostics=details),
                )
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=details)
        else:
            answer = extract_mcq_answer(action)
            reward = float(answer is not None and answer == self.ground_truth)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})
