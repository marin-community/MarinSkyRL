"""Pure computations for rollout and matched-evaluation analysis."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .traces import TraceRecord

CONTEXT_BOUNDS = (0, 16_384, 32_768, 65_536, 131_072)


def comparison_validity(before: list[TraceRecord], after: list[TraceRecord]) -> dict[str, int | bool]:
    """Return whether evaluation traces have task IDs suitable for a delta."""
    common = {record.task_id for record in before} & {record.task_id for record in after}
    return {"common_task_count": len(common), "invalid_for_comparison": not bool(common)}


def matched_reward_delta(before: list[TraceRecord], after: list[TraceRecord]) -> float | None:
    """Return the mean post-minus-baseline reward for common, scored tasks."""
    before_by_task = {record.task_id: record.reward for record in before if record.reward is not None}
    after_by_task = {record.task_id: record.reward for record in after if record.reward is not None}
    deltas = [
        after_by_task[task_id] - before_by_task[task_id] for task_id in before_by_task.keys() & after_by_task.keys()
    ]
    return sum(deltas) / len(deltas) if deltas else None


def temporal_summary(
    records: list[TraceRecord], bin_hours: float
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Aggregate rollout reward and turns into UTC time bins."""
    if bin_hours <= 0:
        raise ValueError("bin_hours must be positive")
    width = timedelta(hours=bin_hours)
    bins: dict[datetime, list[TraceRecord]] = defaultdict(list)
    for record in records:
        if record.timestamp is None:
            continue
        stamp = record.timestamp.astimezone(timezone.utc)
        seconds = int(stamp.timestamp() // width.total_seconds()) * width.total_seconds()
        bins[datetime.fromtimestamp(seconds, tz=timezone.utc)].append(record)
    result: dict[str, dict[str, float | int | None]] = {}
    for start, members in sorted(bins.items()):
        rewards = [member.reward for member in members if member.reward is not None]
        result[start.isoformat()] = {
            "count": len(members),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "mean_turns": sum(member.turns for member in members) / len(members),
            "error_count": sum(member.error_type is not None for member in members),
        }
    return {"bin_hours": bin_hours, "bins": result}


def context_summary(records: list[TraceRecord]) -> dict[str, dict[str, float | int | None]]:
    """Summarize reward and error rate by input-token context bucket."""
    buckets: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        tokens = record.input_tokens or 0
        lower = max(bound for bound in CONTEXT_BOUNDS if bound <= tokens)
        buckets[f"{lower}+"].append(record)
    result: dict[str, dict[str, float | int | None]] = {}
    for label, members in sorted(buckets.items()):
        rewards = [member.reward for member in members if member.reward is not None]
        result[label] = {
            "count": len(members),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "error_rate": sum(member.error_type is not None for member in members) / len(members),
        }
    return result
