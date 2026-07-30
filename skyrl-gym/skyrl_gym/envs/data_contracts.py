"""Stable dataset-preparation contracts for verifier-backed environments.

An environment's rollout implementation remains permissive: malformed examples
score zero instead of crashing a distributed worker. Builders should use this
module before writing data to normalize ground truth and validate a known-good
and known-bad response against the exact runtime verifier.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from skyrl_gym import error
from skyrl_gym.envs.aime import utils as aime_utils
from skyrl_gym.envs.ifeval import utils as ifeval_utils
from skyrl_gym.envs.registration import spec


NormalizeGroundTruth = Callable[[Any], str]
IsCorrect = Callable[[str, str], bool]


@dataclass(frozen=True)
class VerifierDataContract:
    """The public preparation boundary for one verifier environment."""

    env_id: str
    normalize_ground_truth: NormalizeGroundTruth
    is_correct: IsCorrect
    prompt_instruction: str | None = None

    def validate_example(self, ground_truth: Any, positive_response: str, negative_response: str) -> str:
        """Return normalized ground truth after a two-sided verifier preflight."""
        normalized = self.normalize_ground_truth(ground_truth)
        if not self.is_correct(positive_response, normalized):
            raise ValueError(f"{self.env_id} positive response does not satisfy its verifier.")
        if self.is_correct(negative_response, normalized):
            raise ValueError(f"{self.env_id} negative response unexpectedly satisfies its verifier.")
        return normalized


def _normalize_aime(ground_truth: Any) -> str:
    normalized = aime_utils.normalize_final_answer(str(ground_truth))
    if not normalized:
        raise ValueError("AIME ground_truth must normalize to a non-empty answer.")
    return normalized


def _aime_is_correct(response: str, ground_truth: str) -> bool:
    return bool(aime_utils.compute_score(response, ground_truth)["acc"])


def _normalize_ifeval(ground_truth: Any) -> str:
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise ValueError("IFeval ground_truth must be valid JSON.") from exc
    if not isinstance(ground_truth, Mapping):
        raise ValueError("IFeval ground_truth must be a JSON object or mapping.")
    return ifeval_utils.normalize_ground_truth(ground_truth)


def _ifeval_is_correct(response: str, ground_truth: str) -> bool:
    return bool(ifeval_utils.compute_score(response, ground_truth)["acc"])


CONTRACTS = {
    "aime": VerifierDataContract(
        env_id="aime",
        normalize_ground_truth=_normalize_aime,
        is_correct=_aime_is_correct,
        prompt_instruction=aime_utils.BOXED_ANSWER_INSTRUCTION,
    ),
    "ifeval": VerifierDataContract(
        env_id="ifeval",
        normalize_ground_truth=_normalize_ifeval,
        is_correct=_ifeval_is_correct,
    ),
}


def get_data_contract(env_id: str) -> VerifierDataContract:
    """Return the preparation contract for a registered verifier environment.

    The registration lookup makes misspelled or unavailable environment ids fail
    before a training job starts. A registered environment without a contract is
    intentionally rejected until it defines one.
    """
    try:
        spec(env_id)
    except error.Error as exc:
        raise ValueError(f"Unknown environment id: {env_id!r}.") from exc

    contract = CONTRACTS.get(env_id)
    if contract is None:
        raise ValueError(f"Environment {env_id!r} has no verifier data contract.")
    return contract
