"""Local implementation of the Terminal Pivot string-only verifier."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from jsonschema.exceptions import ValidationError
from openapi_schema_validator import validate


def _terminal_schema(harness: str) -> dict[str, Any]:
    text = {"type": "string"}
    number = {"type": "number"}
    boolean = {"type": "boolean"}
    if harness == "terminus_2":
        fields = {"analysis": text, "plan": text, "task_complete": boolean}
        required = ["analysis", "plan", "commands"]
        command_fields = {"keystrokes": text, "duration": number}
        command_required = ["keystrokes"]
    else:
        fields = {"state_analysis": text, "explanation": text, "is_task_complete": boolean}
        required = ["state_analysis", "explanation", "commands", "is_task_complete"]
        command_fields = {"keystrokes": text, "is_blocking": boolean, "timeout_sec": number}
        command_required = list(command_fields)
    fields["commands"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": command_fields,
            "required": command_required,
            "additionalProperties": False,
        },
    }
    return {"type": "object", "properties": fields, "required": required, "additionalProperties": False}


TERMINAL_SCHEMAS = {name: _terminal_schema(name) for name in ("terminus_1", "terminus_2")}


def grade_terminal_pivot(text: str, record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Check JSON schema, completion, and ordered command similarity without execution.

    The released string-only configuration uses a 0.9 threshold, with an optional
    per-row override. Explanations and command durations do not affect similarity.
    """
    metadata = record.get("metadata") or {}
    expected_text = record.get("expected_answer") or metadata.get("expected_answer")
    if not expected_text:
        return 0.0, {"failure_reason": "expected_answer_invalid"}
    if not text.strip():
        return 0.0, {"failure_reason": "model_output_invalid"}
    text = text.rsplit("</think>", 1)[-1].strip()
    parsed = []
    for value, failure in ((expected_text, "expected_answer_invalid"), (text, "model_output_invalid")):
        try:
            obj = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return 0.0, {"failure_reason": failure}
        if not isinstance(obj, dict):
            return 0.0, {"failure_reason": failure}
        parsed.append(obj)
    reference, candidate = parsed
    harness = metadata.get("harness")
    if harness not in TERMINAL_SCHEMAS:
        return 0.0, {"failure_reason": "unknown_harness"}
    for obj, failure in ((reference, "expected_answer_invalid"), (candidate, "schema_check_failed")):
        try:
            validate(obj, TERMINAL_SCHEMAS[harness])
        except ValidationError:
            return 0.0, {"failure_reason": failure}
    completion_field = "task_complete" if reference.get("task_complete") else "is_task_complete"
    if reference.get(completion_field) and not candidate.get(completion_field):
        return 0.0, {"failure_reason": "task_complete_check_failed"}
    reference_commands = [command["keystrokes"] for command in reference["commands"]]
    candidate_commands = [command["keystrokes"] for command in candidate["commands"]]
    # Distinguish an absent command from a command containing an empty string.
    if bool(reference_commands) != bool(candidate_commands):
        similarity = 0.0
    else:
        similarity = SequenceMatcher(None, "".join(reference_commands), "".join(candidate_commands)).ratio()
    threshold = record.get("threshold")
    if threshold is None:
        threshold = 0.9
    passed = similarity >= threshold
    return float(passed), {
        "failure_reason": "none" if passed else "string_similarity_below_threshold",
        "similarity_score": similarity,
        "threshold": threshold,
    }
