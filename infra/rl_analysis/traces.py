"""Read the portable trace artifacts produced by Harbor and SkyRL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

HARBOR_AGGREGATE_TRIAL_COUNT_KEY = "n_total_trials"


@dataclass(frozen=True)
class TraceRecord:
    """The analysis fields shared by rollout and evaluation traces."""

    task_id: str
    reward: float | None
    timestamp: datetime | None
    turns: int
    peak_prompt_tokens: int | None
    error_type: str | None
    cumulative_input_tokens: int | None = None
    summarization_count: int | None = None


@dataclass(frozen=True)
class TrajectoryFields:
    """Analysis fields derived from one ATIF trajectory."""

    agent_turns: int
    peak_prompt_tokens: int | None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _nested_value(record: dict[str, Any], key: str) -> Any:
    result = record.get("result")
    if isinstance(result, dict) and key in result:
        return result[key]
    return record.get(key)


def _task_id(record: dict[str, Any], fallback_task_id: str) -> str:
    task_name = record.get("task") or record.get("task_name")
    if task_name:
        return str(task_name)
    task_id = record.get("task_id")
    if isinstance(task_id, dict):
        org = task_id.get("org")
        name = task_id.get("name")
        ref = task_id.get("ref")
        if org and name:
            return f"{org}/{name}@{ref}" if ref else f"{org}/{name}"
        if name:
            return f"{name}@{ref}" if ref else str(name)
        return fallback_task_id
    return str(task_id or fallback_task_id)


def _reward(record: dict[str, Any]) -> Any:
    verifier_result = record.get("verifier_result")
    if isinstance(verifier_result, dict):
        rewards = verifier_result.get("rewards")
        return rewards.get("reward") if isinstance(rewards, dict) else None
    return _nested_value(record, "reward")


def _trajectory_fields(trajectory: dict[str, Any] | None) -> TrajectoryFields:
    if trajectory is None:
        return TrajectoryFields(0, None)
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return TrajectoryFields(0, None)
    agent_steps = [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]
    prompt_tokens = []
    for step in agent_steps:
        metrics = step.get("metrics")
        value = _number(metrics.get("prompt_tokens")) if isinstance(metrics, dict) else None
        if value is not None:
            prompt_tokens.append(int(value))
    return TrajectoryFields(len(agent_steps), max(prompt_tokens, default=None))


def trace_record(
    record: dict[str, Any],
    fallback_task_id: str,
    trajectory: dict[str, Any] | None = None,
) -> TraceRecord:
    """Normalize one Harbor/SkyRL result mapping into analysis fields."""
    task_id = _task_id(record, fallback_task_id)
    steps = record.get("steps")
    if isinstance(steps, list):
        turns = sum(1 for step in steps if isinstance(step, dict) and step.get("type") == "assistant")
        turns = turns or len(steps)
    else:
        turns = int(_number(record.get("turns")) or 0)
    trajectory_fields = _trajectory_fields(trajectory)
    if trajectory_fields.agent_turns:
        turns = trajectory_fields.agent_turns
    timestamp = _parse_timestamp(record.get("started_at") or record.get("timestamp") or record.get("date"))
    exception_info = record.get("exception_info")
    harbor_error = exception_info.get("exception_type") if isinstance(exception_info, dict) else None
    error = harbor_error or _nested_value(record, "error_type") or record.get("error")
    agent_result = record.get("agent_result")
    cumulative_input_tokens = None
    summarization_count = None
    if isinstance(agent_result, dict):
        cumulative_input_tokens = int(_number(agent_result.get("n_input_tokens")) or 0) or None
        metadata = agent_result.get("metadata")
        if isinstance(metadata, dict):
            summarization_count = int(_number(metadata.get("summarization_count")) or 0)
    return TraceRecord(
        task_id=task_id,
        reward=_number(_reward(record)),
        timestamp=timestamp,
        turns=turns,
        peak_prompt_tokens=(
            trajectory_fields.peak_prompt_tokens or int(_number(record.get("input_tokens")) or 0) or None
        ),
        error_type=str(error) if error else None,
        cumulative_input_tokens=cumulative_input_tokens,
        summarization_count=summarization_count,
    )


def _result_mappings(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        yield from (item for item in payload if isinstance(item, dict))


def load_trace_records(source: Path) -> list[TraceRecord]:
    """Load JSON/JSONL trace results from a local file or artifact directory."""
    source = source.expanduser().resolve()
    paths = [source] if source.is_file() else sorted(source.rglob("result.json"))
    if not paths and source.is_dir():
        paths = sorted(source.rglob("*.jsonl"))
    records: list[TraceRecord] = []
    for path in paths:
        for mapping in _result_mappings(path):
            trajectory_path = path.parent / "agent" / "trajectory.json"
            trajectory = None
            if trajectory_path.is_file():
                payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
                trajectory = payload if isinstance(payload, dict) else None
            if HARBOR_AGGREGATE_TRIAL_COUNT_KEY in mapping and not trajectory_path.is_file():
                continue
            records.append(trace_record(mapping, path.parent.name, trajectory))
    if paths and not any(record.reward is not None for record in records):
        raise ValueError(f"No scored trace records found in non-empty source {source}")
    return records
