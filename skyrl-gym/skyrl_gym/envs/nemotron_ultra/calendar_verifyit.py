# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate calendar times and intervals into core JSONSchema constraints."""

from __future__ import annotations

import math
import re
from itertools import combinations


def _time_to_minutes(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("clock time must be text")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?\s*(am|pm)?", value.strip())
    if match is None:
        raise ValueError("invalid clock time")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3)
    if suffix is None and match.group(2) is None:
        raise ValueError("24-hour clock requires minutes")
    if not 0 <= minute < 60 or not (1 <= hour <= 12 if suffix else (0 <= hour < 24 or hour == 24 and minute == 0)):
        raise ValueError("clock time outside valid range")
    if suffix:
        hour = hour % 12 + (12 if suffix == "pm" else 0)
    return hour * 60 + minute


def _duration(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("duration must be nonnegative numeric data")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("duration must be finite")


def _event_schema(identifier: str, reference: dict) -> dict:
    _duration(reference["duration"])
    start = {"type": "number", "minimum": _time_to_minutes(reference["min_time"])}
    end = {"type": "number", "maximum": _time_to_minutes(reference["max_time"])}
    if start["minimum"] > end["maximum"]:
        raise ValueError("calendar window is reversed")
    constraint = reference["constraint"]
    if constraint is not None:
        if constraint.startswith("before "):
            end = {"allOf": [end, {"maximum": _time_to_minutes(constraint.removeprefix("before "))}]}
        elif constraint.startswith("after "):
            start = {"allOf": [start, {"minimum": _time_to_minutes(constraint.removeprefix("after "))}]}
        elif constraint.startswith("at "):
            start = {"allOf": [start, {"const": _time_to_minutes(constraint.removeprefix("at "))}]}
        elif constraint.startswith("between "):
            lower, upper = constraint.removeprefix("between ").split(" and ")
            low, high = _time_to_minutes(lower), _time_to_minutes(upper)
            if low > high:
                raise ValueError("calendar constraint is reversed")
            start = {"allOf": [start, {"minimum": low}]}
            end = {"allOf": [end, {"maximum": high}]}
        else:
            raise ValueError("invalid calendar constraint")
    return {
        "type": "object",
        "required": ["id", "start", "end", "duration"],
        "properties": {
            "id": {"const": identifier},
            "start": start,
            "end": end,
            "duration": {"type": "number", "minimum": 0, "const": reference["duration"]},
        },
    }


def _calendar(response: str, expected: dict):
    from verifyit.grade import InvalidTask
    from verifyit.json_objects import unique_object
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate
    from skyrl_gym.envs.nemotron_ultra.calendar import _extract_json_list

    try:
        if not isinstance(expected, dict) or any(not isinstance(key, str) for key in expected):
            raise ValueError("calendar references require text identifiers")
        contracts = [_event_schema(key, value) for key, value in expected.items()]
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise InvalidTask("invalid calendar reference") from error
    schema = {"type": "object", "properties": {"think": {"const": False}}}
    instance = {"think": "<think>" in response}
    if expected:
        schema["properties"]["events"] = {
            "type": "array",
            "minItems": len(expected),
            "maxItems": len(expected),
            "items": {"oneOf": contracts},
            "allOf": [
                {
                    "contains": {"type": "object", "properties": {"id": {"const": key}}, "required": ["id"]},
                    "minContains": 1,
                    "maxContains": 1,
                }
                for key in expected
            ],
        }
        schema["properties"]["gaps"] = {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["forward", "backward"],
                "anyOf": [
                    {"properties": {direction: {"type": "number", "minimum": 0}}}
                    for direction in ("forward", "backward")
                ],
            },
        }
        try:
            events = _extract_json_list(response, object_pairs_hook=unique_object)
        except ValueError:
            events = None
        prepared = []
        # Retain one overflow item so core maxItems rejects oversized schedules.
        for event in (events or [])[: len(expected) + 1]:
            try:
                start = _time_to_minutes(event["start_time"])
                prepared.append(
                    {
                        "id": str(event["event_id"]),
                        "start": start,
                        "end": start + event["duration"],
                        "duration": event["duration"],
                    }
                )
            except (KeyError, TypeError, ValueError):
                prepared.append({})
        gaps = []
        for first, second in combinations(prepared, 2):
            try:
                gaps.append({"forward": second["start"] - first["end"], "backward": first["start"] - second["end"]})
            except (KeyError, TypeError):
                gaps.append({})
        instance.update(events=prepared, gaps=gaps)
    return grade_json_schema_candidate(schema, instance)


def grade_calendar_verifyit(response: str, expected: dict) -> tuple[float, str]:
    from verifyit.bounded import call_bounded

    try:
        verdict = call_bounded(_calendar, response, expected, timeout=5)
    except OSError as error:
        raise RuntimeError("calendar verification failed") from error
    path = verdict.detail.get("path", "")
    reason = "pass" if verdict.detail.get("reason") == "valid" else "constraint_violated"
    if path == "think":
        reason = "think_found"
    elif path.startswith("gaps"):
        reason = "conflicting_events"
    elif path == "events":
        reason = "different_number_of_events"
    return verdict.reward, reason
