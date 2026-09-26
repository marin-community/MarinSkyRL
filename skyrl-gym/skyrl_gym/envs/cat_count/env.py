"""One-turn counting environment for the CatCountCanary learning gate."""

from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.cat_count.reward import CatCountScore, cat_count_score
from skyrl_gym.verification import UNKNOWN_STOP_REASON, RolloutEvidence, VerificationResult


class CatCountEnv(BaseTextEnv):
    """Score the decoded assistant turn against the row's requested cat count."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any]):
        super().__init__()
        self.n = int(extras["extra_info"]["n"])
        self.stop_reason = UNKNOWN_STOP_REASON
        self.score: CatCountScore | None = None

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.stop_reason = evidence.stop_reason

    def step(self, action: str) -> BaseTextEnvStepOutput:
        score = cat_count_score(action, self.n, stop_reason=self.stop_reason)
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
        score = self.score
        return {
            "exact": float(score.exact),
            "has_cat": float(score.cat_unigram_count > 0),
            "junk": float(score.junk_words),
            "truncated": float(score.truncated),
            f"exact_n{self.n}": float(score.exact),
            f"n_words_n{self.n}": float(score.n_words),
        }
