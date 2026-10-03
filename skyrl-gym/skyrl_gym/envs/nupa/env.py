"""Single-turn NUPA environment graded with the NUPA5K-Loose policy-eval metric."""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ground_truth_from_extras
from skyrl_gym.envs.nupa.answers import parse_ground_truth
from skyrl_gym.envs.nupa.verifier import NUPAVerifier
from skyrl_gym.verification import RolloutEvidence, VerificationStatus

logger = logging.getLogger(__name__)

METRIC_KEYS = ("exact_match", "digit_match", "dlength", "format_valid", "no_answer")


class NUPAEnv(BaseTextEnv):
    """Score a numeric reasoning response with the NUPA-Loose exact-match metric."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        del env_config
        self._evidence: RolloutEvidence | None = None
        self.verifier: NUPAVerifier | None = None
        try:
            ground_truth = ground_truth_from_extras(extras or {})
            parse_ground_truth(ground_truth)
            self.verifier = NUPAVerifier(ground_truth=ground_truth)
        except (ValueError, AssertionError) as error:
            # Malformed rows score zero instead of crashing a distributed worker;
            # builders reject them earlier through the nupa data contract.
            logger.exception("nupa: invalid ground truth %r; scoring 0.", error)

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self._evidence = evidence

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.verifier is None:
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={})
        evidence = self._evidence or RolloutEvidence(response=action)
        verification = self.verifier.verify(evidence)
        if verification.status is not VerificationStatus.VERIFIED:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"verifier_error": verification.reason},
                verification=verification,
            )
        metadata = {
            "acc": verification.passed is True,
            "pred": verification.diagnostics.get("prediction"),
            **{key: verification.diagnostics[key] for key in METRIC_KEYS},
        }
        return BaseTextEnvStepOutput(
            observations=[],
            reward=verification.score or 0.0,
            done=True,
            metadata=metadata,
            verification=verification,
        )

    @staticmethod
    def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
        aggregated = {"nupa/acc": _mean(metrics, "acc")}
        aggregated.update({f"nupa/{key}": _mean(metrics, key) for key in METRIC_KEYS})
        return aggregated


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else 0.0
