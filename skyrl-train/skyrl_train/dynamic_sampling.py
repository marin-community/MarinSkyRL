"""Shared group-selection policy for dynamic sampling."""

from __future__ import annotations

from enum import StrEnum
from typing import Mapping, Protocol, Sequence


class DynamicSamplingType(StrEnum):
    FILTER = "filter"
    REPLACE = "replace"


class GroupSelectionResult(StrEnum):
    KEEP = "keep"
    UNIFORM_OUTCOMES = "uniform_outcomes"


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
    trajectory_batch: Mapping[str, object], row_indices: Sequence[int] | None = None
) -> bool:
    """Return whether a group has more than one final verifier outcome."""
    response_ids = trajectory_batch.get("response_ids")
    if not isinstance(response_ids, Sequence) or isinstance(response_ids, (str, bytes)):
        raise ValueError("response_ids must be a sequence")
    row_count = len(response_ids)
    if row_indices is None:
        row_indices = range(row_count)
    outcomes = _aligned_sequence(trajectory_batch, "unshaped_rewards", row_count)
    if outcomes is None:
        raise ValueError("dynamic sampling filter requires unshaped_rewards for every generated group")
    is_last_step = _aligned_sequence(trajectory_batch, "is_last_step", row_count)

    final_outcomes = [
        float(outcomes[index])
        for index in row_indices
        if is_last_step is None or bool(is_last_step[index])
    ]
    if not final_outcomes:
        raise ValueError("dynamic sampling group must contain at least one final trial row")
    if len(final_outcomes) == 1:
        return True
    return any(outcome != final_outcomes[0] for outcome in final_outcomes[1:])


class GroupSelectionPolicy:
    """Apply prompt-consuming selection rules after training eligibility checks."""

    def __init__(self, sampling_type: DynamicSamplingType | None) -> None:
        self.sampling_type = sampling_type

    @classmethod
    def for_fully_async(cls, sampling_type: str | None) -> GroupSelectionPolicy:
        resolved_type = DynamicSamplingType(sampling_type) if sampling_type is not None else None
        if resolved_type not in (None, DynamicSamplingType.FILTER):
            raise ValueError(
                "fully asynchronous training supports dynamic_sampling.type=filter or null; "
                f"got {sampling_type!r}"
            )
        return cls(resolved_type)

    def evaluate(self, group: GeneratedGroup) -> GroupSelectionResult:
        if self.sampling_type is None:
            return GroupSelectionResult.KEEP
        if group_is_informative_for_dynamic_sampling(group.trajectory_batch):
            return GroupSelectionResult.KEEP
        return GroupSelectionResult.UNIFORM_OUTCOMES
