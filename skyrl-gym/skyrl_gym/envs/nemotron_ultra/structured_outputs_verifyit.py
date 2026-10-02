"""Client framing for the pinned OpenAPI structured-output contract."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from harbor_config.errors import ErrorCategory, error_category
from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.modes.grade_json_schema import grade_json_schema_candidate

from jsonschema.validators import validates, validator_for
from openapi_schema_validator import OAS32Validator
from referencing import Registry

from .structured_outputs import _coerce_csv, _coerce_xml, _parse, _tool_payload

_DIALECT = "urn:marin:skyrl:openapi-0.9-local-only"
_ORIGINAL_DIALECT = "x-marin-original-schema-dialect"


def _original_schema(schema: dict[str, Any]) -> dict[str, Any]:
    original = dict(schema)
    dialect = original.pop(_ORIGINAL_DIALECT)
    if dialect is None:
        original.pop("$schema", None)
    else:
        original["$schema"] = dialect
    return original


@validates(_DIALECT)
class LocalOpenAPIValidator:
    """Register source-owned validator construction with the existing schema mode."""

    META_SCHEMA = dict(OAS32Validator.META_SCHEMA, **{"$id": _DIALECT})
    ID_OF = staticmethod(OAS32Validator.ID_OF)

    @staticmethod
    def check_schema(schema: dict[str, Any]) -> None:
        OAS32Validator.check_schema(_original_schema(schema))

    def __new__(cls, schema: dict[str, Any]):
        original = _original_schema(schema)
        return OAS32Validator(original, registry=Registry()).evolve(schema=original)


class StructuredOutputPolicy(StrEnum):
    SOURCE = "nemotron_structured_output_source_v1"


@dataclass(frozen=True)
class StructuredOutputInputs:
    text: str
    record: dict[str, Any]
    assistant_message: dict[str, Any]


def structure_structured_output(
    text: str, record: dict[str, Any], assistant_message: dict[str, Any]
) -> StructuredOutputInputs:
    """Snapshot all source fields before selecting or coercing candidate data."""
    try:
        record = deepcopy(record)
    except RecursionError as error:
        raise InvalidTask("Structured-output record exceeds nesting limit") from error
    try:
        return StructuredOutputInputs(text, record, deepcopy(assistant_message))
    except RecursionError as error:
        raise ValueError("Structured-output candidate exceeds nesting limit") from error


def _result(
    verdict: Reward, preparation: dict[str, Any], error_type: str | None = None
) -> tuple[float, dict[str, Any]]:
    # Validator errors can quote protected schema literals, names and property paths.
    detail = {
        key: value
        for key, value in verdict.detail.items()
        if key in {"reason", "passed", "total", "missing", "category", "stage", "source_status", "finalization_policy"}
    }
    return verdict.reward, {
        "error_type": error_type,
        "error_message": error_type,
        "status": verdict.status.value,
        "preparation": preparation,
        "verdict": {"reward": verdict.reward, "status": verdict.status.value, "detail": detail},
    }


def _failure(
    error_type: str, message: str, status: Status, preparation: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    category = (
        ErrorCategory.AGENT
        if status is Status.SCORED
        else error_category("InvalidTask" if status is Status.INVALID_TASK else "RuntimeError")
    )
    verdict = finalize_preparation_failure(
        status=status, category=category, error_type=error_type, message=message, stage="structured_output_policy"
    )
    return _result(verdict, preparation, error_type)


def grade_structured_output_verifyit(
    text: str,
    record: dict[str, Any],
    assistant_message: dict[str, Any],
    *,
    policy: str = StructuredOutputPolicy.SOURCE,
) -> tuple[float, dict[str, Any]]:
    """Apply the named source policy, grading name and payload exclusively in core."""
    preparation = {"policy": str(policy), "provenance_encoding": "python_repr_utf8_v1"}
    candidate_error = None
    try:
        inputs = structure_structured_output(text, record, assistant_message)
        record, assistant_message = inputs.record, inputs.assistant_message
        preparation.update(
            {
                f"{field}_sha256": hashlib.sha256(repr(getattr(inputs, field)).encode()).hexdigest()
                for field in ("text", "record", "assistant_message")
            }
        )
    except InvalidTask as error:
        return _failure("schema_error", str(error), Status.INVALID_TASK, preparation)
    except ValueError as error:
        candidate_error = str(error)
    try:
        preparation["policy"] = StructuredOutputPolicy(policy).value
        for field in ("tool_name", "tool_payload_key"):
            if record.get(field) is not None and not isinstance(record[field], str):
                raise ValueError(f"{field} must be a string or null")
        schema = json.loads(record["schema_str"])
        if not isinstance(schema, dict) or _ORIGINAL_DIALECT in schema:
            raise ValueError("Expected a schema object without reserved client fields")
        if "$schema" in schema:
            selected = validator_for(schema, default=None)
            if selected is None or selected.__module__ not in {
                "jsonschema.validators",
                "openapi_schema_validator.validators",
            }:
                raise ValueError("Unsupported schema dialect")
        OAS32Validator.check_schema(schema)
        framed = dict(schema)
        framed[_ORIGINAL_DIALECT] = framed.get("$schema")
        framed["$schema"] = _DIALECT
        grade_json_schema_candidate(framed, None)
    except Exception as error:
        return _failure("schema_error", str(error)[:200], Status.INVALID_TASK, preparation)

    if candidate_error is not None:
        return _failure("parse_error", candidate_error, Status.SCORED, preparation)

    try:
        components = []
        if record.get("response_mode", "text") == "tool_call":
            calls = assistant_message.get("tool_calls") or []
            if not isinstance(calls, list) or any(
                not isinstance(call, dict) or not isinstance(call.get("function") or {}, dict) for call in calls
            ):
                return _failure("tool_arguments_parse_error", "Malformed tool call", Status.SCORED, preparation)
            # Source cardinality errors precede tool-name validation.
            if len(calls) == 1 and record.get("tool_name"):
                name = (calls[0].get("function") or {}).get("name")
                named = grade_json_schema_candidate({"const": record["tool_name"]}, name)
                components.append(named)
                preparation["effective_tool_name"] = name
                if named.status is not Status.SCORED or named.reward == 0.0:
                    verdict = aggregate_rewards(components, expected_total=2, policy=Aggregation.ALL)
                    return _result(
                        verdict,
                        preparation,
                        "wrong_tool_name" if verdict.status is Status.SCORED else "verification_error",
                    )
            # Name correctness is already core-owned; retain source argument decoding,
            # object admission and payload-key selection without a local name check.
            try:
                value, error_type, message = _tool_payload({**record, "tool_name": None}, assistant_message)
            except (ValueError, RecursionError) as error:
                return _failure("tool_arguments_parse_error", str(error), Status.SCORED, preparation)
            if error_type is not None:
                return _failure(error_type, message, Status.SCORED, preparation)
        else:
            if not text.strip():
                return _failure("empty_response", "No assistant response text", Status.SCORED, preparation)
            schema_type = str(record.get("schema_type", "json")).lower()
            preparation["schema_type"] = schema_type
            try:
                value = _parse(schema_type, text)
            except Exception as error:
                return _failure(
                    "parse_error", f"{type(error).__name__}: {str(error)[:200]}", Status.SCORED, preparation
                )
            preparation["decoded_repr"] = repr(value)
            if schema_type == "xml":
                value = _coerce_xml(value, schema)
            if schema_type == "csv":
                value = _coerce_csv(value, schema)
        preparation["effective_repr"] = repr(value)
        payload = grade_json_schema_candidate(framed, value)
        components.append(payload)
        verdict = aggregate_rewards(components, expected_total=len(components), policy=Aggregation.ALL)
        error_type = None
        if verdict.status is not Status.SCORED:
            error_type = "verification_error"
        elif payload.reward == 0.0:
            error_type = "validation_error"
        return _result(verdict, preparation, error_type)
    except InvalidTask as error:
        return _failure("schema_error", str(error)[:200], Status.INVALID_TASK, preparation)
    except Exception as error:
        return _failure(
            "verification_error", f"{type(error).__name__}: {str(error)[:200]}", Status.INFRA_ERROR, preparation
        )
