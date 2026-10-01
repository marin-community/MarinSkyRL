"""Grade external judge responses with existing schema and exact primitives."""

import json
import math

from verifyit.grade import InvalidTask, Status
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.spec import ExactSpec


def literal_score(candidate: str, reference: str) -> float:
    result = grade_exact_candidate(
        ExactSpec(
            expected=(reference,),
            ignore_case=False,
            ignore_whitespace=False,
            strip_outer_whitespace=False,
        ),
        candidate,
    )
    if result.status is not Status.SCORED:
        raise RuntimeError("Exact judge comparison failed")
    return result.reward


def completion_text(response) -> str:
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Judge must return one complete response")
    choice = choices[0]
    message = choice["message"]
    if (
        choice.get("finish_reason") != "stop"
        or message.get("tool_calls")
        or message.get("refusal")
    ):
        raise ValueError("Judge response is incomplete")
    text = message.get("content")
    if not isinstance(text, str):
        raise ValueError("Judge response has no text")
    return text


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Judge JSON has duplicate keys")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError("Judge JSON must contain finite numbers")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Judge JSON must contain finite numbers")
    return number


def structured_score(raw: str, schema: dict) -> float:
    instance = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )
    validity = grade_json_schema_candidate(schema, instance)
    if validity.status is not Status.SCORED:
        raise InvalidTask("Invalid judge response schema")
    if not validity.reward:
        raise ValueError("Judge response violates its schema")
    return literal_score(instance["correct"], "yes")


def require_reference(question, reference):
    if not isinstance(question, str) or not question.strip():
        raise InvalidTask("Judge question must be nonempty text")
    if not isinstance(reference, str) or not reference.strip():
        raise InvalidTask("Judge reference must be nonempty text")
