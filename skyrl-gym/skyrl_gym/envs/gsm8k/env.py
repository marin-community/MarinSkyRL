from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
    ground_truth_from_extras,
    verification_error_step,
)
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
        self.verifyit_timeout = env_config.get("verifyit_timeout", 10.0)
        self.stop_reason = None
        self.structured_chat = env_config.get("structured_chat", False)

    def init(self, prompt: ConversationType) -> tuple[ConversationType, Dict[str, Any]]:
        metadata = {}
        if self.structured_chat:
            metadata["chat_completion_params"] = {}
        return prompt, metadata

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.stop_reason = evidence.stop_reason

    def _get_reward(self, action: str) -> float:
        if self.reward_method == "final_line" and self.stop_reason not in utils.COMPLETED_STOP_REASONS:
            return 0.0
        return utils.compute_score(
            action, self.ground_truth, method=self.reward_method, verifyit_enabled=self.verifyit_enabled
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        if not self.verifyit_enabled:
            return BaseTextEnvStepOutput(observations=[], reward=self._get_reward(action), done=done, metadata={})
        try:
            from verifyit.grade import Status
            from skyrl_gym.envs.math_verifyit import MathPolicy, grade_math_response

            policy = {
                "strict": MathPolicy.GSM_STRICT,
                "flexible": MathPolicy.GSM_FLEXIBLE,
                "final_line": MathPolicy.GSM_COMPLETED_FINAL_LINE,
            }.get(self.reward_method, self.reward_method)
            verdict = grade_math_response(
                action, self.ground_truth, policy=policy, stop_reason=self.stop_reason, timeout=self.verifyit_timeout
            )
            if verdict.status is not Status.SCORED:
                return verification_error_step(
                    verdict.detail.get("error", "Math verification failed"),
                    minimum_reward=0.0,
                    diagnostics=verdict.detail,
                )
            verification = VerificationResult.verified(
                verdict.reward, passed=verdict.reward == 1.0, diagnostics=verdict.detail
            )
            return BaseTextEnvStepOutput(
                observations=[], reward=verdict.reward, done=True, metadata={}, verification=verification
            )
        except Exception as error:
            return verification_error_step(
                str(error),
                minimum_reward=0.0,
                diagnostics={"verifyit_status": "infra_error", "error_type": type(error).__name__},
            )
