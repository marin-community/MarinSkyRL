"""Bounded opt-in MCQ preparation with source-specific extraction policies."""

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import logging
import re
from typing import Any

from harbor_config.errors import error_category
from verifyit.execution.worker import call_bounded
from verifyit.grade import InvalidTask, Status, finalize_preparation_failure
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_mcq import grade_mcq_candidate
from verifyit.spec import EmptyOutputPolicy, ExactSpec, McqSpec

from skyrl_gym.envs.mcq.utils import extract_mcq_answer
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer
from skyrl_gym.envs.nemotron_ultra.mcqa import (
    ANSWER_COLON_MD_PATTERN,
    ANSWER_COLON_PATTERN,
    _normalize,
    _normalize_extracted,
    _strict_boxed,
    _strip_latex,
)


class MCQPolicy(StrEnum):
    FIRST_BOX = "skyrl_mcq_first_box_v1"
    ULTRA = "skyrl_ultra_mcqa_source_v1"


@dataclass(frozen=True)
class MCQInputs:
    candidate: str
    record: dict[str, Any]


def structure_mcq(candidate: str, record: dict[str, Any]) -> MCQInputs:
    """Capture raw strings and a detached record before extraction or normalization."""
    if not isinstance(record, dict):
        raise InvalidTask("MCQ record must be an object")
    if not isinstance(candidate, str):
        raise InvalidTask("MCQ candidate must be text")
    return MCQInputs(candidate, deepcopy(record))


def _option_matches(value: str, candidate: str) -> bool:
    # Source lower(), unlike casefold(), preserves the source's Unicode contract.
    return bool(
        grade_exact_candidate(
            ExactSpec(
                expected=(_normalize(value),),
                ignore_case=False,
                ignore_whitespace=False,
                strip_outer_whitespace=False,
                empty_output=EmptyOutputPolicy.GRADE,
            ),
            _normalize(candidate),
        ).reward
    )


def prepare_mcq(inputs: MCQInputs, policy: MCQPolicy) -> tuple[McqSpec, str, dict[str, Any]]:
    """Validate the trusted task, then apply its named source extraction policy."""
    record = inputs.record
    gold = record.get("expected_answer")
    if not isinstance(gold, str):
        raise InvalidTask("MCQ expected answer must be text")
    gold = gold.strip().upper()
    if policy is MCQPolicy.FIRST_BOX:
        allowed = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        options = []
        patterns = []
        mode = "first_box"
    elif policy is MCQPolicy.ULTRA:
        options = record.get("options")
        if not isinstance(options, list) or any(
            not isinstance(option, dict)
            or any(
                not isinstance(key, str)
                or len(key) != 1
                or not key.isalpha()
                or (value is not None and not isinstance(value, str))
                for key, value in option.items()
            )
            for option in options
        ):
            raise InvalidTask("MCQA options must contain letter-to-text objects")
        allowed = sorted({key.upper() for option in options for key, value in option.items() if value is not None})
        mode = record.get("grading_mode", "strict_single_letter_boxed")
        if mode not in (
            "strict_single_letter_boxed",
            "lenient_boxed",
            "lenient_answer_colon",
            "lenient_answer_colon_md",
        ):
            raise InvalidTask("Unsupported MCQA grading mode")
        template = record.get("template_metadata")
        if template is not None and not isinstance(template, dict):
            raise InvalidTask("MCQA template_metadata must be an object")
        patterns = [] if template is None else template.get("output_regex", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or any(not isinstance(pattern, str) for pattern in patterns):
            raise InvalidTask("MCQA output_regex must be a string or list of strings")
    else:
        raise InvalidTask("Unsupported MCQ preparation policy")
    if gold not in allowed or not 1 <= len(allowed) <= 26:
        raise InvalidTask("MCQA reference is outside declared options")
    try:
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    except (re.error, OverflowError) as error:
        raise InvalidTask("Malformed MCQA output_regex") from error
    mapping = {letter: chr(ord("A") + index) for index, letter in enumerate(allowed)}
    spec = McqSpec(expected=mapping[gold], options=len(allowed))
    prediction = None
    text = inputs.candidate
    if policy is MCQPolicy.FIRST_BOX:
        prediction = extract_mcq_answer(text)
    else:
        text = final_answer_text(text).strip()
        for pattern in compiled:
            matches = pattern.findall(text)
            if not matches:
                continue
            value = matches[-1]
            if isinstance(value, tuple):
                captures = [capture for capture in value if capture]
                if len(captures) != 1:
                    raise InvalidTask("MCQA output_regex must have one unambiguous answer capture")
                value = captures[0]
            captured = _normalize_extracted(value.strip()).upper()
            if len(captured) == 1 and captured.isalpha():
                prediction = captured
            else:
                prediction = next(
                    (
                        key.upper()
                        for option in options
                        for key, value in option.items()
                        if value is not None and key.upper() in allowed and _option_matches(value, captured)
                    ),
                    None,
                )
            if prediction is not None:
                break
        if prediction is None and mode in ("strict_single_letter_boxed", "lenient_boxed"):
            prediction = _strict_boxed(text, set(allowed))
            boxed = last_boxed_answer(text)
            if prediction is None and mode == "lenient_boxed" and boxed is not None:
                matches = {
                    key.upper()
                    for option in options
                    for key, value in option.items()
                    if value is not None
                    and key.upper() in allowed
                    and any(_option_matches(value, candidate) for candidate in (boxed, _strip_latex(boxed)))
                }
                prediction = next(iter(matches)) if len(matches) == 1 else None
        elif prediction is None and mode == "lenient_answer_colon":
            if match := ANSWER_COLON_PATTERN.search(text):
                candidate = _strip_latex(match.group(1)).strip()
                if len(candidate) == 1 and candidate.upper() in allowed:
                    prediction = candidate.upper()
                else:
                    # Source keeps the first match in each object, then the last matching object.
                    for option in options:
                        for key, value in option.items():
                            if value is not None and key.upper() in allowed and _option_matches(value, candidate):
                                prediction = key.upper()
                                break
        elif prediction is None and mode == "lenient_answer_colon_md":
            if match := ANSWER_COLON_MD_PATTERN.search(text):
                candidate = match.group(1).upper()
                prediction = candidate if candidate in allowed else None
    provenance = {
        "policy": policy.value,
        "grading_mode": mode,
        "output_regex": patterns,
        "option_mapping": mapping,
        "options": options,
        "raw_sha256": hashlib.sha256(
            json.dumps(inputs.__dict__, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    return (
        spec,
        mapping.get(prediction, ""),
        {
            "expected_answer": gold,
            "extracted_answer": prediction,
            "preparation": provenance,
        },
    )


def _grade_mcq(candidate: str, record: dict[str, Any], policy: MCQPolicy) -> tuple[float, dict[str, Any]]:
    spec, answer, details = prepare_mcq(structure_mcq(candidate, record), policy)
    verdict = grade_mcq_candidate(spec, answer)
    return verdict.reward, {**details, "verifyit_status": verdict.status.value}


def grade_mcq(
    candidate: str, record: dict[str, Any], *, policy: MCQPolicy, timeout: float = 10.0
) -> tuple[float, dict[str, Any]]:
    """Bound raw capture, trusted validation, extraction and core scoring together."""
    try:
        reward, details = call_bounded(_grade_mcq, candidate, record, policy, timeout=timeout)
        details["preparation"]["timeout_seconds"] = timeout
        return reward, details
    except Exception as error:
        status = Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR
        if status is Status.INFRA_ERROR:
            logging.getLogger(__name__).exception("MCQ verification failed")
        verdict = finalize_preparation_failure(
            status=status,
            category=error_category(type(error).__name__),
            error_type=type(error).__name__,
            message=str(error),
            stage="mcq_preparation",
        )
        return 0.0, {
            "error_type": "schema_error" if status is Status.INVALID_TASK else "verification_error",
            "verifyit_status": verdict.status.value,
            "error_message": str(error),
            "cause_error_type": type(error).__name__,
            "preparation_stage": "mcq_preparation",
            "preparation": {"policy": policy.value, "timeout_seconds": timeout},
        }
