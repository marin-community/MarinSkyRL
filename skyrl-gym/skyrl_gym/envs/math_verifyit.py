"""Source math preparation and primitive grading under one total deadline."""

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import json

from harbor_config.errors import error_category
from verifyit.adapters.skyrl import (
    grade_aime_extracted,
    grade_gsm8k_extracted,
    grade_gsm8k_final_line,
    grade_literal_candidate,
)
from verifyit.execution.worker import call_bounded
from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.modes.grade_json_schema import grade_json_schema_candidate

from skyrl_gym.envs.aime.utils import extract_minerva_answers, extract_strict_box, normalize_final_answer
from skyrl_gym.envs.gsm8k.utils import extract_solution


class MathPolicy(StrEnum):
    AIME_MINERVA = "skyrl_aime_last300_minerva_v1"
    AIME_BOX = "skyrl_aime_last300_last100_box_v1"
    GSM_STRICT = "skyrl_gsm8k_first_marker_v1"
    GSM_FLEXIBLE = "skyrl_gsm8k_last_number_v1"
    GSM_FINAL_LINE = "skyrl_gsm8k_final_line_decimal_v1"
    GSM_COMPLETED_FINAL_LINE = "skyrl_gsm8k_completed_final_line_decimal_v1"


@dataclass(frozen=True)
class MathInputs:
    response: object
    reference: str
    stop_reason: str | None


def structure_math(response: object, reference: str, stop_reason: str | None) -> MathInputs:
    """Retain the raw immutable fields before any source extraction policy."""
    if not isinstance(reference, str):
        raise InvalidTask("Math reference must be text")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise ValueError("Math stop reason must be text or null")
    return MathInputs(response, reference, stop_reason)


def prepare_math(inputs: MathInputs, policy: MathPolicy) -> tuple[str, str | None]:
    """Validate trusted answers, then apply the selected source extraction policy."""
    if not isinstance(policy, MathPolicy):
        raise InvalidTask("Unsupported math preparation policy")
    reference = inputs.reference
    if policy is MathPolicy.AIME_MINERVA:
        grade_aime_extracted(normalize_final_answer(reference), "")
    elif policy is MathPolicy.AIME_BOX:
        grade_literal_candidate(reference, "")
    else:
        grade_gsm8k_extracted(reference, "")
    if inputs.response is not None and not isinstance(inputs.response, str):
        raise ValueError("Math response must be text or null")
    response = inputs.response or ""
    if policy is MathPolicy.AIME_MINERVA:
        return extract_minerva_answers(response[-300:], reference)
    if policy is MathPolicy.AIME_BOX:
        return reference, extract_strict_box(response[-300:])
    if policy is MathPolicy.GSM_COMPLETED_FINAL_LINE and inputs.stop_reason not in {
        "stop",
        "complete",
        "eos",
        "end_turn",
    }:
        response = ""
    method = {MathPolicy.GSM_STRICT: "strict", MathPolicy.GSM_FLEXIBLE: "flexible"}.get(policy, "final_line")
    return reference, extract_solution(response, method=method)


def _grade_math(response: str | None, reference: str, stop_reason: str | None, policy: MathPolicy) -> Reward:
    inputs = structure_math(response, reference, stop_reason)
    expected, prediction = prepare_math(inputs, policy)
    if policy is MathPolicy.AIME_MINERVA:
        verdict = grade_aime_extracted(expected, prediction or "")
    elif policy is MathPolicy.AIME_BOX:
        verdict = aggregate_rewards(
            (
                grade_literal_candidate(expected, prediction or ""),
                grade_json_schema_candidate({"type": "string"}, prediction),
            ),
            expected_total=2,
            policy=Aggregation.ALL,
        )
    elif policy in (MathPolicy.GSM_FINAL_LINE, MathPolicy.GSM_COMPLETED_FINAL_LINE):
        verdict = grade_gsm8k_final_line(expected, "" if prediction is None else f"#### {prediction}")
    else:
        verdict = grade_gsm8k_extracted(expected, prediction or "")
    return Reward(
        verdict.reward,
        verdict.status,
        {
            "prediction": prediction,
            "verifyit_status": verdict.status.value,
            "preparation": {
                "policy": policy.value,
                "raw_sha256": hashlib.sha256(json.dumps(asdict(inputs), sort_keys=True).encode()).hexdigest(),
            },
        },
    )


def grade_math_response(
    response: str | None,
    reference: str,
    *,
    policy: MathPolicy,
    stop_reason: str | None = None,
    timeout: float = 10.0,
) -> Reward:
    """Bound raw capture, extraction and grading; preserve failures at zero credit."""
    try:
        verdict = call_bounded(_grade_math, response, reference, stop_reason, policy, timeout=timeout)
    except Exception as error:
        verdict = finalize_preparation_failure(
            status=Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR,
            category=error_category(type(error).__name__),
            error_type=type(error).__name__,
            message=str(error),
            stage="math_preparation_and_grading",
        )
    return Reward(
        verdict.reward,
        verdict.status,
        {
            **verdict.detail,
            "verifyit_status": verdict.status.value,
            "preparation": {**verdict.detail.get("preparation", {}), "policy": str(policy), "timeout_seconds": timeout},
        },
    )
