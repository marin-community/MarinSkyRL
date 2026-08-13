"""Typed access to fields in Harbor ``result.json`` artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MISSING_TOKEN_COUNT = -1


@dataclass(frozen=True)
class HarborResult:
    """Fields shared by Harbor reporting and behavioral analysis."""

    task_name: str | None
    trial_name: str | None
    n_episodes: int | None
    n_input_tokens: int | None
    n_output_tokens: int | None
    summarization_count: int | None
    exception_type: str | None
    reward: float | None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def trajectory_count_sequence(trajectory: dict[str, Any]) -> list[tuple[int, int]]:
    """Return token counts for non-copied agent steps, using ``-1`` for missing counts."""
    sequence = []
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict) or step.get("source") != "agent" or step.get("is_copied_context"):
            continue
        metrics = step.get("metrics") or {}
        prompt_tokens = metrics.get("prompt_tokens")
        completion_tokens = metrics.get("completion_tokens")
        parsed_prompt_tokens = _integer(prompt_tokens)
        parsed_completion_tokens = _integer(completion_tokens)
        sequence.append(
            (
                parsed_prompt_tokens if parsed_prompt_tokens is not None else MISSING_TOKEN_COUNT,
                parsed_completion_tokens if parsed_completion_tokens is not None else MISSING_TOKEN_COUNT,
            )
        )
    return sequence


def parse_harbor_result(data: dict[str, Any]) -> HarborResult:
    """Parse the stable analysis fields from one Harbor result mapping."""
    agent_result = data.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    exception_info = data.get("exception_info") or {}
    verifier_result = data.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    reward = rewards.get("reward")
    return HarborResult(
        task_name=_string(data.get("task_name")),
        trial_name=_string(data.get("trial_name")),
        n_episodes=_integer(metadata.get("n_episodes")),
        n_input_tokens=_integer(agent_result.get("n_input_tokens")),
        n_output_tokens=_integer(agent_result.get("n_output_tokens")),
        summarization_count=_integer(metadata.get("summarization_count")),
        exception_type=_string(exception_info.get("exception_type")),
        reward=float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None,
    )
