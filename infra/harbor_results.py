"""Typed access to fields in Harbor ``result.json`` artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    reward: int | float | None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def parse_harbor_result(data: dict[str, Any]) -> HarborResult:
    """Parse the stable analysis fields from one Harbor result mapping."""
    agent_result = data.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    exception_info = data.get("exception_info") or {}
    verifier_result = data.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    reward = rewards.get("reward")
    return HarborResult(
        task_name=data.get("task_name") if isinstance(data.get("task_name"), str) else None,
        trial_name=data.get("trial_name") if isinstance(data.get("trial_name"), str) else None,
        n_episodes=_integer(metadata.get("n_episodes")),
        n_input_tokens=_integer(agent_result.get("n_input_tokens")),
        n_output_tokens=_integer(agent_result.get("n_output_tokens")),
        summarization_count=_integer(metadata.get("summarization_count")),
        exception_type=(
            exception_info.get("exception_type") if isinstance(exception_info.get("exception_type"), str) else None
        ),
        reward=reward if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None,
    )
