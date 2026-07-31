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
from skyrl_gym.envs.gsm8k import utils as gsm8k_utils
from skyrl_gym.envs.ifeval import utils as ifeval_utils
from skyrl_gym.envs.lcb.livecodebench import (
    compute_score,
    normalize_lcb_ground_truth,
)
from skyrl_gym.envs.registration import spec

NormalizeGroundTruth = Callable[[Any], str]
IsCorrect = Callable[[str, str], bool]
LCB_PROMPT_INSTRUCTION = "\nReturn the complete Python solution in this format:\n```python\n# solution\n```"


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


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def _normalize_gsm8k(ground_truth: Any) -> str:
    normalized = str(ground_truth).strip().replace(",", "").replace("$", "")
    if not normalized:
        raise ValueError("gsm8k ground_truth must be non-empty.")
    return normalized


def _gsm8k_is_correct(response: str, ground_truth: str) -> bool:
    return gsm8k_utils.compute_score(response, ground_truth) > 0


def _code_is_correct(response: str, ground_truth: str) -> bool:
    _, reward = compute_score(response, json.loads(ground_truth))
    return reward == 1.0


# ---------------------------------------------------------------------------
# MCQ (multiple choice exact letter match via \boxed{X})
# ---------------------------------------------------------------------------


def _normalize_mcq(ground_truth: Any) -> str:
    answer = str(ground_truth).strip().upper()
    if not answer:
        raise ValueError("mcq ground_truth must be non-empty.")
    return answer


def _mcq_is_correct(response: str, ground_truth: str) -> bool:
    import re

    match = re.search(r"\\boxed\{([A-Za-z])\}", response)
    return match is not None and match.group(1).upper() == ground_truth


# ---------------------------------------------------------------------------
# Preference (schema-only — reward comes from a RM at runtime)
# ---------------------------------------------------------------------------


def _normalize_preference(ground_truth: Any) -> str:
    if not ground_truth:
        raise ValueError("preference ground_truth (chosen response) must be non-empty.")
    return str(ground_truth)


def _preference_is_correct(response: str, ground_truth: str) -> bool:
    return response.strip() == ground_truth.strip()


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
    "gsm8k": VerifierDataContract(
        env_id="gsm8k",
        normalize_ground_truth=_normalize_gsm8k,
        is_correct=_gsm8k_is_correct,
        prompt_instruction="\nThe final answer must appear after ####",
    ),
    "lcb": VerifierDataContract(
        env_id="lcb",
        normalize_ground_truth=normalize_lcb_ground_truth,
        is_correct=_code_is_correct,
        prompt_instruction=LCB_PROMPT_INSTRUCTION,
    ),
    "mcq": VerifierDataContract(
        env_id="mcq",
        normalize_ground_truth=_normalize_mcq,
        is_correct=_mcq_is_correct,
        prompt_instruction="\nAnswer with the option letter from the given choices. Put your answer in \\boxed{ANSWER}",
    ),
    "preference": VerifierDataContract(
        env_id="preference",
        normalize_ground_truth=_normalize_preference,
        is_correct=_preference_is_correct,
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
