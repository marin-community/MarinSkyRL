"""Pure computations for rollout and matched-evaluation analysis."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .traces import TraceRecord

CONTEXT_BOUNDS = (0, 16_384, 32_768, 65_536, 131_072)


def comparison_validity(before: list[TraceRecord], after: list[TraceRecord]) -> dict[str, int | bool]:
    """Return whether evaluation traces have task IDs suitable for a delta."""
    common = {record.task_id for record in before if record.reward is not None} & {
        record.task_id for record in after if record.reward is not None
    }
    return {
        "common_task_count": len(common),
        "baseline_trial_count": sum(record.task_id in common and record.reward is not None for record in before),
        "post_trial_count": sum(record.task_id in common and record.reward is not None for record in after),
        "invalid_for_comparison": not bool(common),
    }


def matched_reward_delta(before: list[TraceRecord], after: list[TraceRecord]) -> float | None:
    """Return the mean post-minus-baseline reward for common, scored tasks."""
    return matched_reward_statistics(before, after)["task_weighted_mean_reward_delta"]


def matched_reward_statistics(before: list[TraceRecord], after: list[TraceRecord]) -> dict[str, int | float | None]:
    """Return replicate-preserving task- and trial-weighted reward deltas."""

    before_by_task: dict[str, list[float]] = defaultdict(list)
    after_by_task: dict[str, list[float]] = defaultdict(list)
    for record in before:
        if record.reward is not None:
            before_by_task[record.task_id].append(record.reward)
    for record in after:
        if record.reward is not None:
            after_by_task[record.task_id].append(record.reward)
    common_tasks = sorted(before_by_task.keys() & after_by_task.keys())
    deltas = [
        sum(after_by_task[task_id]) / len(after_by_task[task_id])
        - sum(before_by_task[task_id]) / len(before_by_task[task_id])
        for task_id in common_tasks
    ]
    before_trials = [reward for task_id in common_tasks for reward in before_by_task[task_id]]
    after_trials = [reward for task_id in common_tasks for reward in after_by_task[task_id]]
    return {
        "common_task_count": len(common_tasks),
        "baseline_trial_count": len(before_trials),
        "post_trial_count": len(after_trials),
        "task_weighted_mean_reward_delta": sum(deltas) / len(deltas) if deltas else None,
        "trial_weighted_mean_reward_delta": (
            sum(after_trials) / len(after_trials) - sum(before_trials) / len(before_trials)
            if before_trials and after_trials
            else None
        ),
    }


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
        total_input_tokens = [member.total_input_tokens for member in members if member.total_input_tokens is not None]
        summarization_counts = [
            member.summarization_count for member in members if member.summarization_count is not None
        ]
        result[start.isoformat()] = {
            "count": len(members),
            "mean_reward": sum(rewards) / len(rewards) if rewards else None,
            "mean_turns": sum(member.turns for member in members) / len(members),
            "mean_total_input_tokens": (
                sum(total_input_tokens) / len(total_input_tokens) if total_input_tokens else None
            ),
            "mean_summarization_count": (
                sum(summarization_counts) / len(summarization_counts) if summarization_counts else None
            ),
            "error_count": sum(member.error_type is not None for member in members),
        }
    return {"bin_hours": bin_hours, "bins": result}


def context_summary(records: list[TraceRecord]) -> dict[str, dict[str, float | int | None]]:
    """Summarize reward and error rate by peak request prompt-token bucket."""
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
