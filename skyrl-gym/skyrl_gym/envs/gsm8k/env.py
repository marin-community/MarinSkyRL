from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ground_truth_from_extras
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any
from omegaconf import DictConfig
from skyrl_gym.verification import RolloutEvidence, VerificationResult


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        self.ground_truth = ground_truth_from_extras(extras)
        self.reward_method = env_config.get("reward_method", "strict")
        self.verifyit_enabled = bool(env_config.get("verifyit_enabled", False))
        self.stop_reason = None

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.stop_reason = evidence.stop_reason

    def _get_reward(self, action: str) -> float:
        if self.verifyit_enabled:
            from verifyit.adapters.skyrl import grade_gsm8k_extracted

            grade_gsm8k_extracted(self.ground_truth, "")
        if self.reward_method == "final_line" and self.stop_reason not in {"stop", "complete", "eos", "end_turn"}:
            return 0.0
        return utils.compute_score(
            action, self.ground_truth, method=self.reward_method, verifyit_enabled=self.verifyit_enabled
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        if self.verifyit_enabled:
            from verifyit.grade import InvalidTask

            try:
                reward = self._get_reward(action)
            except InvalidTask as error:
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={},
                    verification=VerificationResult.error(str(error), diagnostics={"verifyit_status": "invalid_task"}),
                )
        else:
            reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
