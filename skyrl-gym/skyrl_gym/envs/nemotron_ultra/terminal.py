# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminus-2 string-only verification from NVIDIA-NeMo/Gym resources_servers/terminus_judge.

Matches the released string-only configuration: schema, completion state, and
SequenceMatcher over concatenated keystrokes with a default threshold of 0.9.
The optional hybrid verifier accepts string matches or JEV action equivalence.
No shell commands are executed.
"""

import json
from difflib import SequenceMatcher
from typing import Any

from jsonschema import ValidationError, validate

from skyrl_gym.envs.nemotron_ultra.terminal_judge import TerminalJudge

TERMINUS_2_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis", "plan", "commands"],
    "properties": {
        "analysis": {"type": "string"},
        "plan": {"type": "string"},
        "task_complete": {"type": "boolean"},
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["keystrokes"],
                "properties": {"keystrokes": {"type": "string"}, "duration": {"type": "number"}},
            },
        },
    },
}
TERMINAL_PIVOT_VERIFIERS = ("exact_commands", "string_90", "schema_completion")


def grade_terminal_verifiers(
    text: str, record: dict[str, Any], *, judge: TerminalJudge | None = None
) -> dict[str, Any]:
    """Score Terminal JSON actions with exact, released string, and schema/flag checks."""
    if record["metadata"].get("harness") != "terminus_2":
        raise ValueError("The pinned Terminal release requires the terminus_2 harness")
    verifiers = (*TERMINAL_PIVOT_VERIFIERS, "string_or_jev") if judge is not None else TERMINAL_PIVOT_VERIFIERS
    expected = json.loads(record["expected_answer"])
    validate(expected, TERMINUS_2_SCHEMA)
    try:
        candidate = json.loads(text.rsplit("</think>", 1)[-1].strip())
        validate(candidate, TERMINUS_2_SCHEMA)
    except (json.JSONDecodeError, ValidationError):
        return {"scores": dict.fromkeys(verifiers, 0.0), "reason": "invalid_response"}
    if expected.get("task_complete", False) and not candidate.get("task_complete", False):
        return {"scores": dict.fromkeys(verifiers, 0.0), "reason": "task_incomplete"}
    expected_commands, commands = expected["commands"], candidate["commands"]
    similarity = (
        0.0
        if bool(expected_commands) != bool(commands)
        else SequenceMatcher(
            None,
            "".join(item["keystrokes"] for item in expected_commands),
            "".join(item["keystrokes"] for item in commands),
        ).ratio()
    )
    threshold = record.get("threshold")
    grades = {
        "scores": {
            "exact_commands": float(
                [item["keystrokes"] for item in expected_commands] == [item["keystrokes"] for item in commands]
            ),
            "string_90": float(similarity >= (0.9 if threshold is None else threshold)),
            "schema_completion": 1.0,
        },
        "similarity": similarity,
    }
    if judge is not None:
        passed = grades["scores"]["string_90"]
        if not passed:
            grades["jev"] = judge.grade(expected, candidate)
            passed = float(grades["jev"]["passed"])
        grades["scores"]["string_or_jev"] = passed
    return grades


def grade_terminal(text: str, record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Return the released Terminal agent's binary reward and diagnostics."""
    grades = grade_terminal_verifiers(text, record)
    return grades["scores"]["string_90"], {key: value for key, value in grades.items() if key != "scores"}
