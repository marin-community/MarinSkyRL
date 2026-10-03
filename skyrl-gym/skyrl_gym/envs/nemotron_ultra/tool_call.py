# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict Ultra tool-call grading and released Pivot action comparison."""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from typing import Any


class StepRewardCategory(StrEnum):
    NO_ACTION_FOUND = "No tool call or chat message was found in the response"
    NO_EXPECTED_TOOL_CALL = "No tool call was found when one was expected"
    EXPECTED_CHAT_MESSAGE_FOUND = "A chat message was found as expected"
    NO_EXPECTED_CHAT_MESSAGE = "A tool call was executed when a chat message was expected"
    UNEXPECTED_TOOL = "The tool in a tool call is not the expected tool"
    ARGUMENTS_DECODE_ERROR = "An error occurred when decoding the arguments string in a tool call as a JSON object"
    ARGUMENT_VALUE_TYPE_DIFFERENT = "The type of an argument value in a tool call is different than the expected type"
    ARGUMENT_OBJECT_KEYS_DIFFERENT = (
        "The keys in an object in an argument value in a tool call are different than the keys in the expected object"
    )
    ARGUMENT_LIST_LENGTH_DIFFERENT = (
        "A list in an argument value in a tool call has a different length than the expected list"
    )
    ARGUMENT_VALUE_DIFFERENT = "An argument value in a tool call is different than the expected value"
    EXPECTED_TOOL_CALL = "A tool call that matches the expected tool call was found"
    MISSING_EXPECTED_CALLS = "The response does not contain distinct matches for all expected calls"
    UNEXPECTED_CALL_COUNT = "The response must contain exactly one tool call"


def _compare_arguments(
    expected: Any,
    actual: Any,
    *,
    floating_point_comparison_threshold: float = 1e-6,
) -> tuple[bool, StepRewardCategory | None]:
    if not isinstance(actual, type(expected)):
        return False, StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return False, StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT
        for key, value in expected.items():
            matches, category = _compare_arguments(
                value,
                actual[key],
                floating_point_comparison_threshold=floating_point_comparison_threshold,
            )
            if not matches:
                return matches, category
        return True, None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False, StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT
        for expected_item, actual_item in zip(expected, actual):
            matches, category = _compare_arguments(
                expected_item,
                actual_item,
                floating_point_comparison_threshold=floating_point_comparison_threshold,
            )
            if not matches:
                return matches, category
        return True, None
    if isinstance(expected, float):
        if abs(actual - expected) < floating_point_comparison_threshold:
            return True, None
        return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
    if expected == actual:
        return True, None
    return False, StepRewardCategory.ARGUMENT_VALUE_DIFFERENT


def grade_expected_action(
    expected_action: dict[str, Any],
    assistant_message: dict[str, Any],
) -> tuple[float, StepRewardCategory]:
    """Return the exact binary reward and diagnostic category used by NeMo Gym."""
    tool_calls = assistant_message.get("tool_calls") or []
    content = assistant_message.get("content")
    expected_type = expected_action.get("type")
    if expected_type == "message":
        if isinstance(content, str) and content.strip() and not tool_calls:
            return 1.0, StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND
        return 0.0, StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE
    if expected_type != "function_call":
        raise NotImplementedError(f"Unsupported expected action type {expected_type!r}")
    if not tool_calls:
        return 0.0, (
            StepRewardCategory.NO_EXPECTED_TOOL_CALL if isinstance(content, str) else StepRewardCategory.NO_ACTION_FOUND
        )

    if len(tool_calls) != 1:
        return 0.0, StepRewardCategory.NO_EXPECTED_TOOL_CALL

    actual = tool_calls[0].get("function") or {}
    if expected_action.get("name") != actual.get("name"):
        return 0.0, StepRewardCategory.UNEXPECTED_TOOL
    try:
        expected_arguments = json.loads(expected_action["arguments"])
        actual_arguments = json.loads(actual.get("arguments"))
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        return 0.0, StepRewardCategory.ARGUMENTS_DECODE_ERROR
    matches, category = _compare_arguments(
        expected_arguments,
        actual_arguments,
    )
    if matches:
        return 1.0, StepRewardCategory.EXPECTED_TOOL_CALL
    assert category is not None
    return 0.0, category


def _argument_mismatch(expected: Any, actual: Any, threshold: float) -> StepRewardCategory | None:
    pending = [(expected, actual)]
    while pending:
        reference, candidate = pending.pop()
        # Preserve the reference verifier's isinstance semantics, including bool/int asymmetry.
        if not isinstance(candidate, type(reference)):
            return StepRewardCategory.ARGUMENT_VALUE_TYPE_DIFFERENT
        if isinstance(reference, dict):
            if reference.keys() != candidate.keys():
                return StepRewardCategory.ARGUMENT_OBJECT_KEYS_DIFFERENT
            pending.extend((value, candidate[key]) for key, value in reversed(reference.items()))
        elif isinstance(reference, list):
            if len(reference) != len(candidate):
                return StepRewardCategory.ARGUMENT_LIST_LENGTH_DIFFERENT
            pending.extend(reversed(list(zip(reference, candidate))))
        elif isinstance(reference, float):
            if not abs(reference - candidate) < 1e-6:
                return StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
        elif isinstance(reference, str):
            left, right = Counter(reference.lower().split()), Counter(candidate.lower().split())
            if min(left.total(), right.total()) < 2:
                matches = reference == candidate
            else:
                # The published score has maximum 0.5; it is not a Jaccard index.
                matches = (left & right).total() / (left.total() + right.total()) >= threshold
            if not matches:
                return StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
        elif reference != candidate:
            return StepRewardCategory.ARGUMENT_VALUE_DIFFERENT
    return None


def _assign_call(expected_index: int, edges: list[list[int]], owners: dict[int, int], visited: set[int]) -> bool:
    for candidate_index in edges[expected_index]:
        if candidate_index in visited:
            continue
        visited.add(candidate_index)
        if candidate_index not in owners or _assign_call(owners[candidate_index], edges, owners, visited):
            owners[candidate_index] = expected_index
            return True
    return False


def grade_pivot_action(
    expected_action: dict[str, Any],
    assistant_message: dict[str, Any],
    *,
    word_count_similarity_threshold: float = 0.1,
) -> tuple[float, StepRewardCategory]:
    """Grade a Chat Completions assistant message using the Pivot dataset defaults.

    Calls take precedence over text. Expected calls need distinct, unordered
    matches; extra generated calls are allowed. Invalid reference arguments raise
    a data error, while invalid generated arguments cannot match.
    """
    actual = [call.get("function", {}) for call in assistant_message.get("tool_calls") or []]
    content = assistant_message.get("content")
    has_text = isinstance(content, str)
    if not actual and not has_text:
        return 0.0, StepRewardCategory.NO_ACTION_FOUND
    if expected_action["type"] == "message":
        return (
            (0.0, StepRewardCategory.NO_EXPECTED_CHAT_MESSAGE)
            if actual
            else (1.0, StepRewardCategory.EXPECTED_CHAT_MESSAGE_FOUND)
        )
    if expected_action["type"] == "function_call":
        expected = [expected_action]
    elif expected_action["type"] == "function_call_batch":
        expected = expected_action["calls"]
        if not expected:
            raise ValueError("An expected function-call batch must be nonempty")
    else:
        raise ValueError(f"Unsupported expected action type: {expected_action['type']!r}")
    expected_arguments = [json.loads(call["arguments"]) for call in expected]
    if not actual:
        return 0.0, StepRewardCategory.NO_EXPECTED_TOOL_CALL

    edges: list[list[int]] = []
    mismatch = StepRewardCategory.MISSING_EXPECTED_CALLS
    for reference, arguments in zip(expected, expected_arguments):
        matches = []
        for index, candidate in enumerate(actual):
            if reference["name"] != candidate.get("name"):
                mismatch = StepRewardCategory.UNEXPECTED_TOOL
                continue
            try:
                parsed = json.loads(candidate.get("arguments"))
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                mismatch = StepRewardCategory.ARGUMENTS_DECODE_ERROR
                continue
            reason = _argument_mismatch(arguments, parsed, word_count_similarity_threshold)
            if reason is None:
                matches.append(index)
            else:
                mismatch = reason
        edges.append(matches)
    owners: dict[int, int] = {}
    for index in range(len(expected)):
        if not _assign_call(index, edges, owners, set()):
            return 0.0, mismatch if len(expected) == len(actual) == 1 else StepRewardCategory.MISSING_EXPECTED_CALLS
    return 1.0, StepRewardCategory.EXPECTED_TOOL_CALL


PIVOT_VERIFIERS = ("tool_name", "nemo", "exact")


def grade_pivot_verifiers(expected_action: dict[str, Any], assistant_message: dict[str, Any]) -> dict[str, Any]:
    """Score one saved or online response with the three frozen pilot verifiers."""
    calls = assistant_message.get("tool_calls") or []
    if expected_action["type"] not in {"message", "function_call"}:
        raise ValueError("The SWE pilot requires a single demonstrated action")
    if expected_action["type"] == "function_call" and len(calls) != 1:
        return {
            "scores": dict.fromkeys(PIVOT_VERIFIERS, 0.0),
            "categories": dict.fromkeys(PIVOT_VERIFIERS, StepRewardCategory.UNEXPECTED_CALL_COUNT.value),
            "tool_call_count": len(calls),
            "extra_tool_calls": max(0, len(calls) - 1),
        }
    names = (
        calls[0].get("function", {}).get("name") == expected_action["name"]
        if expected_action["type"] == "function_call"
        else not calls
    )
    exact, exact_category = grade_expected_action(expected_action, assistant_message)
    nemo, nemo_category = grade_pivot_action(expected_action, assistant_message, word_count_similarity_threshold=0.0)
    malformed = 0
    for call in calls:
        try:
            json.loads(call.get("function", {}).get("arguments"))
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            malformed += 1
    return {
        "scores": {"tool_name": float(names), "nemo": nemo, "exact": exact},
        "categories": {"exact": exact_category.value, "nemo": nemo_category.value},
        "malformed_tool_calls": malformed,
        "tool_call_count": len(calls),
        "extra_tool_calls": max(0, len(calls) - int(expected_action["type"] == "function_call")),
    }
