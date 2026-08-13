"""Read the portable trace artifacts produced by Harbor and SkyRL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from infra.harbor_results import (
    HARBOR_TRAJECTORY_PATH,
    MISSING_TOKEN_COUNT,
    HarborResult,
    parse_harbor_result,
    trajectory_count_sequence,
)

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


def _trajectory_fields(trajectory: dict[str, Any]) -> TrajectoryFields:
    token_counts = trajectory_count_sequence(trajectory)
    prompt_tokens = [count for count, _ in token_counts if count != MISSING_TOKEN_COUNT]
    return TrajectoryFields(len(token_counts), max(prompt_tokens, default=None))


def _turn_count(record: dict[str, Any], harbor: HarborResult, trajectory: TrajectoryFields) -> int:
    if trajectory.agent_turns:
        return trajectory.agent_turns
    steps = record.get("steps")
    if isinstance(steps, list):
        assistant_turns = sum(1 for step in steps if isinstance(step, dict) and step.get("type") == "assistant")
        return assistant_turns or len(steps)
    return harbor.n_episodes or int(_number(record.get("turns")) or 0)


def trace_record(
    record: dict[str, Any],
    fallback_task_id: str,
    trajectory_fields: TrajectoryFields,
) -> TraceRecord:
    """Normalize one Harbor/SkyRL result mapping into analysis fields."""
    task_id = _task_id(record, fallback_task_id)
    harbor = parse_harbor_result(record)
    timestamp = _parse_timestamp(record.get("started_at") or record.get("timestamp") or record.get("date"))
    error = harbor.exception_type or _nested_value(record, "error_type") or record.get("error")
    return TraceRecord(
        task_id=task_id,
        reward=harbor.reward if harbor.reward is not None else _number(_nested_value(record, "reward")),
        timestamp=timestamp,
        turns=_turn_count(record, harbor, trajectory_fields),
        peak_prompt_tokens=trajectory_fields.peak_prompt_tokens,
        error_type=str(error) if error else None,
        cumulative_input_tokens=harbor.n_input_tokens,
        summarization_count=harbor.summarization_count,
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
    """Load JSON/JSONL traces, joining Harbor trajectories and skipping job aggregates.

    A non-empty source with no scored records is rejected so a schema mismatch cannot silently
    produce an empty analysis.
    """
    source = source.expanduser().resolve()
    paths = [source] if source.is_file() else sorted(source.rglob("result.json"))
    if not paths and source.is_dir():
        paths = sorted(source.rglob("*.jsonl"))
    records: list[TraceRecord] = []
    for path in paths:
        for mapping in _result_mappings(path):
            trajectory_path = path.parent / HARBOR_TRAJECTORY_PATH
            trajectory_fields = TrajectoryFields(0, None)
            if trajectory_path.is_file():
                payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected a JSON object in Harbor trajectory {trajectory_path}")
                trajectory_fields = _trajectory_fields(payload)
            if HARBOR_AGGREGATE_TRIAL_COUNT_KEY in mapping and not trajectory_path.is_file():
                continue
            records.append(trace_record(mapping, path.parent.name, trajectory_fields))
    if paths and not any(record.reward is not None for record in records):
        raise ValueError(f"No scored trace records found in non-empty source {source}")
    return records
