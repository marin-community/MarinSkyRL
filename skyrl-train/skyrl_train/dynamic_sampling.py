"""Shared group-selection policy for dynamic sampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import statistics
from typing import Mapping, Protocol, Sequence

from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward


class DynamicSamplingType(StrEnum):
    FILTER = "filter"
    REPLACE = "replace"


class DynamicSamplingRewardSource(StrEnum):
    SHAPED = "shaped"
    UNSHAPED = "unshaped"


DEFAULT_DYNAMIC_SAMPLING_REWARD_SOURCE = DynamicSamplingRewardSource.SHAPED
DEFAULT_DYNAMIC_SAMPLING_MIN_REWARD_STD = 0.0


@dataclass(frozen=True)
class DynamicSamplingCriteria:
    reward_source: DynamicSamplingRewardSource
    min_reward_std: float


def resolve_dynamic_sampling_criteria(
    informative_on: str = DEFAULT_DYNAMIC_SAMPLING_REWARD_SOURCE,
    min_reward_std: float = DEFAULT_DYNAMIC_SAMPLING_MIN_REWARD_STD,
) -> DynamicSamplingCriteria:
    reward_source = DynamicSamplingRewardSource(informative_on)
    if not math.isfinite(min_reward_std) or min_reward_std < 0:
        raise ValueError("dynamic_sampling.min_reward_std must be finite and non-negative")
    return DynamicSamplingCriteria(reward_source=reward_source, min_reward_std=min_reward_std)


class GroupSelectionResult(StrEnum):
    KEEP = "keep"
    INSUFFICIENT_REWARD_SPREAD = "insufficient_reward_spread"


class GeneratedGroup(Protocol):
    trajectory_batch: Mapping[str, object]


def _aligned_sequence(batch: Mapping[str, object], key: str, row_count: int) -> Sequence[object] | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence when present")
    if len(value) != row_count:
        raise ValueError(f"{key} must have one entry per response row, got {len(value)} and {row_count}")
    return value


def group_is_informative_for_dynamic_sampling(
    trajectory_batch: Mapping[str, object],
    row_indices: Sequence[int] | None = None,
    *,
    criteria: DynamicSamplingCriteria,
) -> bool:
    """Return whether a group's configured final rewards have sufficient spread."""
    response_ids = trajectory_batch.get("response_ids")
    if not isinstance(response_ids, Sequence) or isinstance(response_ids, (str, bytes)):
        raise ValueError("response_ids must be a sequence")
    row_count = len(response_ids)
    if row_indices is None:
        row_indices = range(row_count)
    reward_key = "rewards" if criteria.reward_source is DynamicSamplingRewardSource.SHAPED else "unshaped_rewards"
    outcomes = _aligned_sequence(trajectory_batch, reward_key, row_count)
    if outcomes is None:
        raise ValueError(f"dynamic sampling filter requires {reward_key} for every generated group")
    is_last_step = _aligned_sequence(trajectory_batch, "is_last_step", row_count)

    final_outcomes = []
    for index in row_indices:
        if is_last_step is not None and not bool(is_last_step[index]):
            continue
        final_outcomes.append(NormalizedReward.from_output(outcomes[index]).total)
    if not final_outcomes:
        raise ValueError("dynamic sampling group must contain at least one final trial row")
    if len(final_outcomes) == 1:
        return True
    return statistics.pstdev(final_outcomes) > criteria.min_reward_std


class GroupSelectionPolicy:
    """Apply prompt-consuming selection rules after training eligibility checks."""

    def __init__(
        self,
        sampling_type: DynamicSamplingType | None,
        *,
        criteria: DynamicSamplingCriteria,
    ) -> None:
        self.sampling_type = sampling_type
        self.criteria = criteria

    @classmethod
    def for_fully_async(
        cls,
        sampling_type: str | None,
        *,
        informative_on: str = DEFAULT_DYNAMIC_SAMPLING_REWARD_SOURCE,
        min_reward_std: float = DEFAULT_DYNAMIC_SAMPLING_MIN_REWARD_STD,
    ) -> GroupSelectionPolicy:
        resolved_type = DynamicSamplingType(sampling_type) if sampling_type is not None else None
        if resolved_type not in (None, DynamicSamplingType.FILTER):
            raise ValueError(
                f"fully asynchronous training supports dynamic_sampling.type=filter or null; got {sampling_type!r}"
            )
        criteria = resolve_dynamic_sampling_criteria(informative_on, min_reward_std)
        return cls(resolved_type, criteria=criteria)

    def evaluate(self, group: GeneratedGroup) -> GroupSelectionResult:
        if self.sampling_type is None:
            return GroupSelectionResult.KEEP
        if group_is_informative_for_dynamic_sampling(
            group.trajectory_batch,
            criteria=self.criteria,
        ):
            return GroupSelectionResult.KEEP
        return GroupSelectionResult.INSUFFICIENT_REWARD_SPREAD
