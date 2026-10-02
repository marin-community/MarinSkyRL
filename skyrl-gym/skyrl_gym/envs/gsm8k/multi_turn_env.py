from typing import Dict, Any
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, verification_error_step
from skyrl_gym.envs.gsm8k import utils
from skyrl_gym.verification import RewardResult, VerificationResult


class GSM8kMultiTurnEnv(BaseTextEnv):
    """
    Multi-turn GSM8k environment with turn-level rewards.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        reward_spec = extras.get("reward_spec", {})
        assert "ground_truth" in reward_spec, "reward_spec.ground_truth is required"

        self.ground_truth: str = reward_spec["ground_truth"]
        self.verifyit_enabled = bool(env_config.get("verifyit_enabled", False))
        self.verifyit_timeout = env_config.get("verifyit_timeout", 10.0)
        self.max_turns = 5
        if "max_turns" in extras:
            self.max_turns = int(extras["max_turns"])
        elif "max_turns" in extras["extra_info"]:
            self.max_turns = int(extras["extra_info"]["max_turns"])

        format_score = 0.2
        self.format_score_per_turn: float = format_score / self.max_turns

    def init(self, prompt):
        # No special pre-processing; return prompt and empty metadata
        return prompt, {}

    def _make_observation(self) -> list[dict]:
        remaining = self.max_turns - self.turns
        if remaining <= 0:
            return []

        if remaining > 1:
            msg = (
                "Please provide your step-by-step reasoning, "
                "and also include a tentative numeric answer at the end in the exact format: '#### ANSWER'."
            )
        else:
            msg = "Now provide only the final numeric answer in the exact format: '#### ANSWER'."

        return [{"role": "user", "content": msg}]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        verification = None
        reward_result = None
        if self.verifyit_enabled:
            try:
                from verifyit.grade import Status
                from skyrl_gym.envs.math_verifyit import MathPolicy, grade_math_response

                verdict = grade_math_response(
                    action, self.ground_truth, policy=MathPolicy.GSM_STRICT, timeout=self.verifyit_timeout
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
                # The source format bonus is an optimization reward, separate from correctness.
                bonus = (
                    self.format_score_per_turn
                    if verdict.detail["prediction"] is not None and not verdict.reward
                    else 0.0
                )
                reward = verdict.reward + bonus
                reward_result = RewardResult(
                    unshaped_reward=verdict.reward, optimization_reward=reward, components={"format": bonus}
                )
            except Exception as error:
                return verification_error_step(
                    str(error),
                    minimum_reward=0.0,
                    diagnostics={"verifyit_status": "infra_error", "error_type": type(error).__name__},
                )
        else:
            reward = utils.compute_score(
                action, self.ground_truth, method="strict", format_score=self.format_score_per_turn, score=1.0
            )
        done = self.turns >= self.max_turns or reward == 1.0

        observations = [] if done else self._make_observation()

        output = BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={},
        )
        if verification is not None:
            output["verification"] = verification
            output["reward_result"] = reward_result
        return output

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "steps": self.turns,
        }

    @staticmethod
    def aggregate_metrics(metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        n = len(metrics)
        avg_steps = sum(float(m.get("steps", 0)) for m in metrics) / n
        return {"avg_steps": avg_steps}
