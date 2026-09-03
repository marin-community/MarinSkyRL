"""Synchronous rollout-group admission and replacement collection."""

from dataclasses import dataclass
from typing import TypedDict

from skyrl_train.batch_sampling import accumulate_selected_groups
from skyrl_train.group_admission import (
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdvantageInvariant,
    TrainingGroupInvariantError,
)
from skyrl_train.trajectory_runners.base import TrajectoryBatch


class InsufficientEligibleGroupsError(RuntimeError):
    """Synchronous generation exhausted its replacement budget before filling a batch."""


class GroupAdmissionSamplingState(TypedDict, total=False):
    """Accepted synchronous groups collected while replacements are generated."""

    sample_batch_count: int
    collected_trajectory_batch: TrajectoryBatch
    collected_uids: list[str]
    num_prompts_in_batch: int
    rejection_counts: dict[str, int]
    inspected_count: int


@dataclass(frozen=True)
class GroupAdmissionSamplingResult:
    trajectory_batch: TrajectoryBatch
    uids: list[str]
    keep_sampling: bool
    state: GroupAdmissionSamplingState | None
    rejection_counts: dict[AdmissionRejection, int]
    inspected_count: int


def admit_or_collect_replacements(
    trajectory_batch: TrajectoryBatch,
    uids: list[str],
    *,
    invariant: GroupAdvantageInvariant,
    rollout_logprobs_required: bool,
    target_batch_size: int,
    tis_lcs_alert_threshold: float,
    state: GroupAdmissionSamplingState,
    step_wise: bool = False,
) -> GroupAdmissionSamplingResult:
    """Drop retryable groups and collect a complete replacement batch."""
    policy = GroupAdmissionPolicy(
        invariant,
        max_staleness_steps=0,
        rollout_logprobs_required=rollout_logprobs_required,
    )
    admissions = policy.evaluate_batch(trajectory_batch, uids, global_step=0)
    retryable = {AdmissionRejection.FULLY_MASKED, AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE}
    rejection_counts = {rejection: 0 for rejection in AdmissionRejection}
    selected_uids = []
    for admission in admissions:
        if admission.decision.accepted:
            selected_uids.append(admission.uid)
            continue
        assert admission.decision.primary_rejection is not None
        rejection_counts[admission.decision.primary_rejection] += 1
        if any(rejection not in retryable for rejection in admission.decision.rejections):
            raise TrainingGroupInvariantError(
                uid=admission.uid,
                decision=admission.decision,
                physical_count=admission.physical_count,
                expected_physical_count=invariant.physical_group_size,
                row_indices=admission.row_indices,
            )

    accumulated_rejections = state.setdefault("rejection_counts", {})
    for rejection, count in rejection_counts.items():
        accumulated_rejections[rejection.value] = accumulated_rejections.get(rejection.value, 0) + count
    state["inspected_count"] = state.get("inspected_count", 0) + len(admissions)

    if (
        len(selected_uids) == target_batch_size
        and state.get("collected_trajectory_batch") is None
        and not any(accumulated_rejections.values())
    ):
        return GroupAdmissionSamplingResult(
            trajectory_batch=trajectory_batch,
            uids=uids,
            keep_sampling=False,
            state=None,
            rejection_counts={rejection: 0 for rejection in AdmissionRejection},
            inspected_count=state["inspected_count"],
        )

    accumulated = accumulate_selected_groups(
        trajectory_batch,
        uids,
        selected_uids,
        target_group_count=target_batch_size,
        sample_batch_count=state["sample_batch_count"],
        tis_lcs_alert_threshold=tis_lcs_alert_threshold,
        require_rollout_logprobs=rollout_logprobs_required,
        state=state,
        step_wise=step_wise,
    )
    return GroupAdmissionSamplingResult(
        trajectory_batch=accumulated.trajectory_batch,
        uids=accumulated.uids,
        keep_sampling=accumulated.keep_sampling,
        state=state if accumulated.keep_sampling else None,
        rejection_counts={
            rejection: accumulated_rejections.get(rejection.value, 0) for rejection in AdmissionRejection
        },
        inspected_count=state["inspected_count"],
    )
