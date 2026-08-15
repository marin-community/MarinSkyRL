"""Adapters from Harbor result objects to shared verifier contracts."""

from collections.abc import Mapping
from typing import Protocol

from loguru import logger
from skyrl_gym.verification import VerificationResult


class HarborVerifierResult(Protocol):
    rewards: Mapping[str, float]
    stdout: str | None


class HarborExceptionInfo(Protocol):
    exception_type: str


class HarborTrialResult(Protocol):
    verifier_result: HarborVerifierResult | None
    exception_info: HarborExceptionInfo | None


def verification_from_harbor_result(result: HarborTrialResult) -> VerificationResult:
    """Adapt Harbor's nullable verifier result without treating zero as missing."""
    verifier_result = result.verifier_result
    if verifier_result is None:
        exception_info = result.exception_info
        if exception_info is not None:
            exception_type = exception_info.exception_type
            return VerificationResult.error(
                f"verification unavailable after {exception_type}",
                diagnostics={"exception_type": exception_type},
            )
        return VerificationResult.unavailable("Harbor returned no verifier result")

    if not isinstance(verifier_result.rewards, Mapping) or "reward" not in verifier_result.rewards:
        logger.error("Harbor verifier result is malformed: missing rewards['reward']")
        return VerificationResult.error(
            "Harbor verifier result is malformed",
            diagnostics={"missing_field": "rewards.reward"},
        )
    score = verifier_result.rewards["reward"]
    return VerificationResult.verified(
        score,
        passed=float(score) > 0.0,
        diagnostics={"stdout": verifier_result.stdout},
    )
