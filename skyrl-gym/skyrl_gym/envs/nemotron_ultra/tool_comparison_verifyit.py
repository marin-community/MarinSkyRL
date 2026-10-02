"""Translate Nemotron actions into verifyit schema and numeric contracts."""

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from harbor_config.errors import error_category
from verifyit.grade import Aggregation, InvalidTask, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.preparation.errors import InvalidPreparation, PreparationFailure
from verifyit.json_objects import unique_object
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.modes.grade_math import grade_numeric_candidate
from verifyit.spec import NumericSpec

from skyrl_gym.envs.nemotron_ultra.tool_call import StepRewardCategory

FLOAT_THRESHOLD = math.nextafter(1e-6, 0.0)
MESSAGE_SCHEMA = {
    "type": "object",
    "required": ["content", "tool_calls"],
    "properties": {
        "content": {"type": "string", "pattern": r"\S"},
        "tool_calls": {"type": "array", "maxItems": 0},
    },
}


def _object(properties: dict) -> dict:
    return {"type": "object", "required": list(properties), "properties": properties}


def _typed(value: Any, path: tuple = (), numbers: dict | None = None) -> Any:
    """JSON Schema's integer type includes floats; retain Python numeric tags."""
    if type(value) in (int, float):
        if numbers is not None:
            numbers[path] = value
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, dict):
        return {key: _typed(child, (*path, key), numbers) for key, child in value.items()}
    if isinstance(value, list):
        return [_typed(child, (*path, index), numbers) for index, child in enumerate(value)]
    return value


def _schema(value: Any, path: tuple, numeric: dict) -> dict:
    if isinstance(value, dict):
        return {
            "type": "object",
            "required": list(value),
            "additionalProperties": False,
            "properties": {key: _schema(child, (*path, key), numeric) for key, child in value.items()},
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "minItems": len(value),
            "maxItems": len(value),
            **(
                {"prefixItems": [_schema(child, (*path, index), numeric) for index, child in enumerate(value)]}
                if value
                else {}
            ),
        }
    if type(value) in (int, float):
        number = {"type": "number"}
        if type(value) is float:
            numeric[path] = NumericSpec(expected=value, tolerance_abs=FLOAT_THRESHOLD, tolerance_rel=0.0)
        else:
            number["const"] = value
        return {
            "type": "object",
            "required": ["kind", "value"],
            "additionalProperties": False,
            "properties": {"kind": {"const": type(value).__name__}, "value": number},
        }
    return {"type": {str: "string", bool: "boolean", type(None): "null"}[type(value)], "const": value}


def _category(verdict, expected_type: str) -> StepRewardCategory:
    """Translate core diagnostics; this function never determines a score."""
    if expected_type == "message":
        return (
            StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND
            if verdict.detail.get("reason") == "valid"
            else StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE
        )
    path = verdict.detail.get("path", "")
    error = verdict.detail.get("error", "")
    if verdict.detail.get("reason") == "valid":
        return StepRewardCategory.EXPECTED_TOOL_CALL
    if path.endswith("/name"):
        return StepRewardCategory.UNEXPECTED_TOOL
    if path.endswith("/decoded"):
        return StepRewardCategory.ARGUMENTS_DECODE_ERROR
    if "/arguments/value" not in path:
        return StepRewardCategory.NO_EXPECTED_TOOL_CALL
    if path.endswith("/kind") or "not of type" in error:
        return StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT
    if "required property" in error or "Additional properties" in error:
        return StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT
    if "too short" in error or "too long" in error:
        return StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT
    return StepRewardCategory.ARGUMENT_VALUE_DIFFERENT


class ToolComparisonPolicy(StrEnum):
    STRICT_TYPED = "nemotron_strict_typed_arguments_v1"


@dataclass(frozen=True)
class ToolActionInputs:
    expected_action: dict[str, Any]
    assistant_message: dict[str, Any]


@dataclass(frozen=True)
class PreparedToolAction:
    raw: ToolActionInputs
    policy: ToolComparisonPolicy
    expected_type: str
    schema: dict
    instance: dict
    numeric: dict
    actual_numbers: dict


def structure_tool_action(expected_action: dict[str, Any], assistant_message: dict[str, Any]) -> ToolActionInputs:
    """Snapshot the complete source records without selecting or normalizing fields."""
    return ToolActionInputs(deepcopy(expected_action), deepcopy(assistant_message))


def _prepare_tool_action(inputs: ToolActionInputs, policy: ToolComparisonPolicy) -> PreparedToolAction:
    """Apply the established opt-in JSON, numeric-type, and absent-call policies."""
    expected_action, assistant_message = inputs.expected_action, inputs.assistant_message
    if not isinstance(expected_action, dict) or expected_action.get("type") not in ("message", "function_call"):
        raise InvalidTask("expected action must specify message or function_call")
    expected_type = expected_action["type"]
    instance = {
        "content": assistant_message.get("content"),
        "tool_calls": [] if assistant_message.get("tool_calls") is None else assistant_message["tool_calls"],
    }
    if expected_type == "message":
        return PreparedToolAction(inputs, policy, expected_type, deepcopy(MESSAGE_SCHEMA), instance, {}, {})
    name, text = expected_action.get("name"), expected_action.get("arguments")
    if not isinstance(name, str) or not isinstance(text, str):
        raise InvalidTask("expected tool call needs a name and JSON argument string")
    try:
        arguments = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, RecursionError) as error:
        raise InvalidTask("expected tool arguments are not unambiguous JSON") from error
    numeric = {}
    argument_schema = _schema(arguments, (), numeric)
    function_schema = _object(
        {
            "name": {"type": "string", "const": name},
            "arguments": _object({"decoded": {"const": True}, "value": argument_schema}),
        }
    )
    schema = _object(
        {
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": _object({"function": function_schema}),
            }
        }
    )
    calls = instance["tool_calls"]
    actual_numbers = {}
    if isinstance(calls, list):
        prepared = []
        for call in calls:
            if isinstance(call, dict) and isinstance(call.get("function"), dict):
                function = dict(call["function"])
                try:
                    decoded = json.loads(function.get("arguments"), object_pairs_hook=unique_object)
                    function["arguments"] = {"decoded": True, "value": _typed(decoded, numbers=actual_numbers)}
                except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
                    function["arguments"] = {"decoded": False, "value": None}
                call = {**call, "function": function}
            prepared.append(call)
        instance["tool_calls"] = prepared
    return PreparedToolAction(inputs, policy, expected_type, schema, instance, numeric, actual_numbers)


def prepare_tool_action(
    inputs: ToolActionInputs, policy: str = ToolComparisonPolicy.STRICT_TYPED
) -> PreparedToolAction:
    """Prepare atomically; trusted configuration failures finalize at core minimum."""
    try:
        return _prepare_tool_action(inputs, ToolComparisonPolicy(policy))
    except (InvalidTask, ValueError, RecursionError) as error:
        failure = PreparationFailure(
            Status.INVALID_TASK, error_category("InvalidTask"), type(error).__name__, str(error), "tool_policy"
        )
        verdict = finalize_preparation_failure(
            status=failure.status,
            category=failure.category,
            error_type=failure.error_type,
            message=failure.message,
            stage=failure.stage,
        )
        raise InvalidPreparation(failure, verdict) from error


def grade_prepared_tool_action(prepared: PreparedToolAction) -> tuple[float, StepRewardCategory]:
    """Grade the prepared contract using core schema, numeric and ALL operations."""
    expected_type = prepared.expected_type
    numeric = prepared.numeric
    actual_numbers = prepared.actual_numbers
    # Validate trusted contracts even when the candidate contains no usable call.
    for spec in numeric.values():
        grade_numeric_candidate(spec, spec.expected)
    structural = grade_json_schema_candidate(prepared.schema, prepared.instance)
    leaves = [grade_numeric_candidate(spec, actual_numbers.get(path, math.nan)) for path, spec in numeric.items()]
    verdict = aggregate_rewards([structural, *leaves], expected_total=1 + len(leaves), policy=Aggregation.ALL)
    category = _category(structural, expected_type)
    if structural.detail.get("reason") == "valid" and any(leaf.reward == 0.0 for leaf in leaves):
        category = StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
    return verdict.reward, category


def grade_expected_action_verifyit(
    expected_action: dict[str, Any], assistant_message: dict[str, Any]
) -> tuple[float, StepRewardCategory]:
    """Prepare the source records and delegate all grading decisions to core."""
    return grade_prepared_tool_action(prepare_tool_action(structure_tool_action(expected_action, assistant_message)))
