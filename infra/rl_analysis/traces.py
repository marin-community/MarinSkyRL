"""Read the portable trace artifacts produced by Harbor and SkyRL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class TraceRecord:
    """The analysis fields shared by rollout and evaluation traces."""

    task_id: str
    reward: float | None
    timestamp: datetime | None
    turns: int
    input_tokens: int | None
    error_type: str | None


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


def trace_record(record: dict[str, Any], fallback_task_id: str) -> TraceRecord:
    """Normalize one Harbor/SkyRL result mapping into analysis fields."""
    task_id = str(record.get("task") or record.get("task_id") or fallback_task_id)
    steps = record.get("steps")
    if isinstance(steps, list):
        turns = sum(1 for step in steps if isinstance(step, dict) and step.get("type") == "assistant")
        turns = turns or len(steps)
    else:
        turns = int(_number(record.get("turns")) or 0)
    timestamp = _parse_timestamp(record.get("started_at") or record.get("timestamp") or record.get("date"))
    error = _nested_value(record, "error_type") or record.get("error")
    return TraceRecord(
        task_id=task_id,
        reward=_number(_nested_value(record, "reward")),
        timestamp=timestamp,
        turns=turns,
        input_tokens=int(_number(record.get("input_tokens")) or 0) or None,
        error_type=str(error) if error else None,
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
            records.append(trace_record(mapping, path.parent.name))
    return records
