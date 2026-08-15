from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.aime.verifier import AIMERewardPolicy, AIMEVerifier
from skyrl_gym.metrics import default_aggregate_metrics
from skyrl_gym.verification import RolloutEvidence
from typing import Dict, Any
from omegaconf import DictConfig


class AIMEEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.

    Supports an optional, tunable LENGTH-PENALTY reward (cosine length-scaled,
    gated on correctness). All length-penalty knobs are read from `env_config`
    (i.e. hydra `environment.skyrl_gym.aime.*`). The single grid axis is
    `length_penalty_weight`; with weight=0 (the default) the reward is byte-for-byte
    the legacy +1.0/-1.0 reward, so it is a clean A/B superset.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_model" in extras, "reward_model field is required"
        assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
        self.ground_truth = extras["reward_model"]["ground_truth"]

        # ---- Tunable length-penalty config (hydra: environment.skyrl_gym.aime.*) ----
        # weight=0.0 -> legacy reward (backward compatible).
        self.verifier = AIMEVerifier(
            ground_truth=self.ground_truth,
            evaluation_token_budget=int(env_config.get("evaluation_token_budget", 8192)),
        )
        self.reward_policy = AIMERewardPolicy(
            length_penalty_weight=float(env_config.get("length_penalty_weight", 0.0)),
            target_length=int(env_config.get("target_length", 0)),
            max_generation_tokens=int(env_config.get("max_gen_length", 0)),
            truncated_penalty=float(env_config.get("truncated_penalty", -2.0)),
            min_response_length=int(env_config.get("min_response_length", 16)),
        )
        self._evidence: RolloutEvidence | None = None

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self._evidence = evidence

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step

        evidence = self._evidence or RolloutEvidence(response=action)
        verification = self.verifier.verify(evidence)
        reward_result, reward_diagnostics = self.reward_policy.evaluate(evidence, verification)
        metadata = {
            "acc": verification.passed is True,
            "pred": verification.diagnostics["prediction"],
            **verification.diagnostics,
            **reward_diagnostics,
        }

        # No observation in aime, and no tool call
        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward_result.optimization_reward,
            done=done,
            metadata=metadata,
            verification=verification,
            reward_result=reward_result,
        )

    @staticmethod
    def aggregate_metrics(metrics: list[Dict[str, Any]]) -> Dict[str, float]:
        def fraction(rows: list[Dict[str, Any]], key: str) -> float:
            return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0

        correct = [row for row in metrics if bool(row.get("acc"))]
        incorrect = [row for row in metrics if not bool(row.get("acc"))]
        aggregated = default_aggregate_metrics(metrics)
        aggregated.update(
            {
                "over_evaluation_budget_fraction": fraction(metrics, "over_evaluation_budget"),
                "correct_over_evaluation_budget_fraction": fraction(correct, "over_evaluation_budget"),
                "incorrect_over_evaluation_budget_fraction": fraction(incorrect, "over_evaluation_budget"),
                "answered_within_evaluation_budget_fraction": fraction(metrics, "answered_within_evaluation_budget"),
            }
        )
        return aggregated
