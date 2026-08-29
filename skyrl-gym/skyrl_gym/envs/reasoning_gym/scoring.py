"""Reasoning Gym ground-truth normalization and task-native scoring."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import reasoning_gym

ANSWER_MARKER = "Answer:"


def normalize_ground_truth(ground_truth: Any) -> str:
    """Validate and serialize a task name with its complete generated entry."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise ValueError("Reasoning Gym ground_truth must be valid JSON.") from exc
    if not isinstance(ground_truth, Mapping):
        raise TypeError("Reasoning Gym ground_truth must be a mapping.")
    task = ground_truth.get("task")
    entry = ground_truth.get("entry")
    if not isinstance(task, str) or not task:
        raise ValueError("Reasoning Gym ground_truth requires a task name.")
    if not isinstance(entry, Mapping) or not isinstance(entry.get("answer"), str):
        raise ValueError("Reasoning Gym ground_truth requires an entry with a canonical answer.")
    if not isinstance(entry.get("metadata"), Mapping) or entry["metadata"].get("source_dataset") != task:
        raise ValueError("Reasoning Gym entry metadata must match its task name.")
    reasoning_gym.get_score_answer_fn(task)
    return json.dumps({"entry": dict(entry), "task": task}, sort_keys=True)


def extract_answer(response: str) -> str:
    """Return the text after the last ``Answer:`` marker, or the whole response if absent.

    Reasoning Gym verifiers expect the bare answer: scoring a full chain-of-thought
    response yields only length-ratio fuzz credit even when the final answer is correct.
    """
    _, marker, answer = response.rpartition(ANSWER_MARKER)
    return answer.strip() if marker else response.strip()


def score_response(response: str, ground_truth: str) -> float:
    """Score a response's extracted final answer with the generated task's package verifier."""
    spec = json.loads(normalize_ground_truth(ground_truth))
    score = reasoning_gym.get_score_answer_fn(spec["task"])(extract_answer(response), spec["entry"])
    return float(score)
