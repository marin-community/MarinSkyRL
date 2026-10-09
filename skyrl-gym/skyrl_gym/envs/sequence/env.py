"""One-turn environment that scores a reply against the row's expected sequence of items."""

from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.sequence.reward import SequenceScore, sequence_score
from skyrl_gym.verification import UNKNOWN_STOP_REASON, RolloutEvidence, VerificationResult


class SequenceEnv(BaseTextEnv):
    """Score the decoded assistant turn against ``extra_info.items``; ``extra_info.n`` labels the metrics."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any]):
        super().__init__()
        self.items = [str(item) for item in extras["extra_info"]["items"]]
        self.n = int(extras["extra_info"].get("n", len(self.items)))
        self.stop_reason = UNKNOWN_STOP_REASON
        self.score: SequenceScore | None = None

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        return prompt, {}

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.stop_reason = evidence.stop_reason

    def step(self, action: str) -> BaseTextEnvStepOutput:
        score = sequence_score(action, self.items, stop_reason=self.stop_reason)
        self.score = score
        return BaseTextEnvStepOutput(
            observations=[],
            reward=score.reward,
            done=True,
            metadata={},
            verification=VerificationResult.verified(float(score.exact), passed=score.exact),
        )

    def get_metrics(self) -> dict[str, float]:
        if self.score is None:
            return {}
        return {
            "exact": float(self.score.exact),
            "prefix_fraction": self.score.correct_prefix / len(self.items),
            "truncated": float(self.score.truncated),
            f"exact_n{self.n}": float(self.score.exact),
            f"n_words_n{self.n}": float(self.score.n_words),
        }
