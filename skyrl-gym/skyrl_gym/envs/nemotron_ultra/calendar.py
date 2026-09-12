# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calendar constraint reward ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import json
import re
from typing import Any


def _time_to_minutes(value: str) -> int:
    value = value.strip()
    if "am" in value or "pm" in value:
        suffix = "am" if "am" in value else "pm"
        time = value.replace(suffix, "")
        hour, minute = (map(int, time.split(":")) if ":" in time else (int(time), 0))
        return (hour * 60 if hour != 12 else 0) + minute + (12 * 60 if suffix == "pm" else 0)
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _extract_json_list(text: str) -> list[Any] | None:
    pattern = r"\[(?:[^\[\]]|\{[^}]*\})*\{(?:[^\[\]]|\{[^}]*\})*\}(?:[^\[\]]|\{[^}]*\})*\]"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _conflicts(events: list[dict[str, Any]], event: dict[str, Any]) -> bool:
    start = _time_to_minutes(event["start_time"])
    end = start + event["duration"]
    for other in events:
        if other is event:
            continue
        other_start = _time_to_minutes(other["start_time"])
        other_end = other_start + other["duration"]
        if not (end <= other_start or start >= other_end):
            return True
    return False


def _satisfies(event: dict[str, Any], expected: dict[str, Any]) -> bool:
    if event["duration"] != expected["duration"]:
        return False
    start = _time_to_minutes(event["start_time"])
    end = start + event["duration"]
    if start < _time_to_minutes(expected["min_time"]) or end > _time_to_minutes(expected["max_time"]):
        return False
    constraint = expected["constraint"]
    if constraint is None:
        return True
    if constraint.startswith("before "):
        return end <= _time_to_minutes(constraint.removeprefix("before "))
    if constraint.startswith("after "):
        return start >= _time_to_minutes(constraint.removeprefix("after "))
    if constraint.startswith("between "):
        low, high = constraint.removeprefix("between ").split(" and ")
        return start >= _time_to_minutes(low) and end <= _time_to_minutes(high)
    if constraint.startswith("at "):
        return start == _time_to_minutes(constraint.removeprefix("at "))
    return True


def grade_calendar(response: str, expected: dict[str, Any]) -> tuple[float, str]:
    if "<think>" in response:
        return 0.0, "think_found"
    if not expected:
        return 1.0, "pass"
    try:
        events = _extract_json_list(response)
        if not events:
            return 0.0, "no_json_list"
        by_id = {str(event["event_id"]): event for event in events}
        if len(by_id) != len(expected):
            return 0.0, "different_number_of_events"
        if any(_conflicts(events, event) for event in events):
            return 0.0, "conflicting_events"
        if any(not _satisfies(by_id[event_id], expected[event_id]) for event_id in expected):
            return 0.0, "constraint_violated"
    except Exception:
        return 0.0, "error_in_grading"
    return 1.0, "pass"
