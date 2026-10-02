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
        self.ground_truth = str(extras["reward_model"]["ground_truth"]).strip().upper()
        self.verifyit_enabled = bool(env_config.get("verifyit_enabled", False))

    def step(self, action: str) -> BaseTextEnvStepOutput:
        answer = extract_mcq_answer(action)
        if self.verifyit_enabled:
            from verifyit.grade import InvalidTask
            from verifyit.modes.grade_mcq import grade_mcq_candidate
            from verifyit.spec import McqSpec

            try:
                reward = grade_mcq_candidate(McqSpec(expected=self.ground_truth, options=26), answer or "").reward
            except InvalidTask as error:
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={},
                    verification=VerificationResult.error(str(error), diagnostics={"verifyit_status": "invalid_task"}),
                )
        else:
            reward = float(answer is not None and answer == self.ground_truth)
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})
