"""Pure computations for rollout and matched-evaluation analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import fmean
from typing import Sequence

from .traces import TraceRecord

CONTEXT_BOUNDS = (0, 16_384, 32_768, 65_536, 131_072)
UNKNOWN_CONTEXT_BUCKET = "unknown"


def _optional_mean(values: Sequence[float | int]) -> float | None:
    return fmean(values) if values else None


@dataclass(frozen=True)
class MatchedRewardStatistics:
    """Replicate-preserving statistics for common scored tasks."""

    common_task_count: int
    matched_baseline_trial_count: int
    matched_post_trial_count: int
    task_weighted_mean_reward_delta: float | None
    trial_weighted_mean_reward_delta: float | None
    invalid_for_comparison: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "invalid_for_comparison", self.common_task_count == 0)


@dataclass(frozen=True)
class TemporalBin:
    count: int
    mean_reward: float | None
    mean_turns: float
    mean_cumulative_input_tokens: float | None
    mean_summarization_count: float | None
    error_count: int


@dataclass(frozen=True)
class TemporalSummary:
    bin_hours: float
    bins: dict[str, TemporalBin]


@dataclass(frozen=True)
class ContextBin:
    count: int
    mean_reward: float | None
    error_rate: float


def matched_reward_statistics(before: list[TraceRecord], after: list[TraceRecord]) -> MatchedRewardStatistics:
    """Return replicate-preserving task- and trial-weighted reward deltas."""

    def rewards_by_task(records: list[TraceRecord]) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for record in records:
            if record.reward is not None:
                grouped[record.task_id].append(record.reward)
        return grouped

    before_by_task = rewards_by_task(before)
    after_by_task = rewards_by_task(after)
    common_tasks = sorted(before_by_task.keys() & after_by_task.keys())
    deltas = [
        fmean(after_by_task[task_id]) - fmean(before_by_task[task_id])
        for task_id in common_tasks
    ]
    def common_trials(grouped: dict[str, list[float]]) -> list[float]:
        return [reward for task_id in common_tasks for reward in grouped[task_id]]

    before_trials = common_trials(before_by_task)
    after_trials = common_trials(after_by_task)
    return MatchedRewardStatistics(
        common_task_count=len(common_tasks),
        matched_baseline_trial_count=len(before_trials),
        matched_post_trial_count=len(after_trials),
        task_weighted_mean_reward_delta=fmean(deltas) if deltas else None,
        trial_weighted_mean_reward_delta=(
            fmean(after_trials) - fmean(before_trials)
            if before_trials and after_trials
            else None
        ),
    )


def mean_reward(records: list[TraceRecord]) -> float | None:
    """Return the mean over scored records, or ``None`` when none are scored."""
    rewards = [record.reward for record in records if record.reward is not None]
    return _optional_mean(rewards)


def temporal_summary(
    records: list[TraceRecord], bin_hours: float
) -> TemporalSummary:
    """Aggregate rollout reward, turns, errors, and token usage into UTC time bins."""
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
    result: dict[str, TemporalBin] = {}
    for start, members in sorted(bins.items()):
        cumulative_input_tokens = [
            member.cumulative_input_tokens for member in members if member.cumulative_input_tokens is not None
        ]
        summarization_counts = [
            member.summarization_count for member in members if member.summarization_count is not None
        ]
        result[start.isoformat()] = TemporalBin(
            count=len(members),
            mean_reward=mean_reward(members),
            mean_turns=fmean(member.turns for member in members),
            mean_cumulative_input_tokens=_optional_mean(cumulative_input_tokens),
            mean_summarization_count=_optional_mean(summarization_counts),
            error_count=sum(member.error_type is not None for member in members),
        )
    return TemporalSummary(bin_hours=bin_hours, bins=result)


def context_summary(records: list[TraceRecord]) -> dict[str, ContextBin]:
    """Summarize reward and error rate by peak request prompt-token bucket."""
    buckets: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        if record.peak_prompt_tokens is None:
            buckets[UNKNOWN_CONTEXT_BUCKET].append(record)
        else:
            lower = max(bound for bound in CONTEXT_BOUNDS if bound <= record.peak_prompt_tokens)
            buckets[f"{lower}+"].append(record)
    result: dict[str, ContextBin] = {}
    for label, members in sorted(buckets.items()):
        result[label] = ContextBin(
            count=len(members),
            mean_reward=mean_reward(members),
            error_rate=sum(member.error_type is not None for member in members) / len(members),
        )
    return result
