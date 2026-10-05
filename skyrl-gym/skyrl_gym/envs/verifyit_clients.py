"""Opt-in clients for existing verifyit graders; source extraction stays in callers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _grade_reasoning_entry(task: str, entry: dict[str, Any], answer: str) -> float:
    from verifyit.grade import Status, run
    from verifyit.spec import ReasoningGymSpec, render_spec

    with TemporaryDirectory(prefix="skyrl-verifyit-") as directory:
        root = Path(directory)
        (root / "entry.json").write_text(json.dumps(entry, allow_nan=False))
        (root / "answer.txt").write_text(answer)
        spec = root / "verifier.toml"
        spec.write_text(render_spec(ReasoningGymSpec(dataset=task)))
        verdict = run(spec, root)

    if verdict.status is not Status.SCORED or verdict.detail.get("reason") == "scorer_error":
        raise RuntimeError("Reasoning Gym verification failed")
    return verdict.reward


def grade_reasoning_entry(task: str, entry: dict[str, Any], answer: str) -> float:
    try:
        return _grade_reasoning_entry(task, entry, answer)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("Reasoning Gym verification failed") from error
