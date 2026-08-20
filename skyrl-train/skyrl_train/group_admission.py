"""Algorithm-aware admission rules for generated rollout groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Protocol, Sequence


class GroupAdvantageKind(StrEnum):
    """Supported group-relative advantage contracts."""

    EXACT_PHYSICAL = "exact_physical"
    MINIMUM_BASELINE_ELIGIBLE = "minimum_baseline_eligible"
    NONE = "none"


@dataclass(frozen=True)
class GroupAdvantageInvariant:
    """Resolved per-run group contract shared by admission and advantage math."""

    kind: GroupAdvantageKind
    physical_group_size: int
    minimum_group_size: int | None

    def __post_init__(self) -> None:
        if self.physical_group_size < 1:
            raise ValueError(f"physical_group_size must be positive, got {self.physical_group_size}")
        if self.kind is GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE:
            if self.minimum_group_size is None:
                raise ValueError("minimum_baseline_eligible requires minimum_group_size")
            if not 2 <= self.minimum_group_size <= self.physical_group_size:
                raise ValueError(
                    "minimum_group_size must be between 2 and physical_group_size, got "
                    f"{self.minimum_group_size} and {self.physical_group_size}"
                )
        elif self.minimum_group_size is not None:
            raise ValueError(f"{self.kind} does not accept minimum_group_size")

    @classmethod
    def exact_physical(cls, *, physical_group_size: int) -> GroupAdvantageInvariant:
        return cls(GroupAdvantageKind.EXACT_PHYSICAL, physical_group_size, None)

    @classmethod
    def minimum_baseline_eligible(cls, *, physical_group_size: int, minimum_group_size: int) -> GroupAdvantageInvariant:
        return cls(GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE, physical_group_size, minimum_group_size)

    @classmethod
    def no_group_advantage(cls, *, physical_group_size: int) -> GroupAdvantageInvariant:
        return cls(GroupAdvantageKind.NONE, physical_group_size, None)

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> GroupAdvantageInvariant:
        return cls(
            kind=GroupAdvantageKind(str(config["kind"])),
            physical_group_size=int(config["physical_group_size"]),
            minimum_group_size=(
                int(config["minimum_group_size"]) if config.get("minimum_group_size") is not None else None
            ),
        )

    def to_config(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "physical_group_size": self.physical_group_size,
            "minimum_group_size": self.minimum_group_size,
        }


class AdmissionRejection(StrEnum):
    """Retryable reasons that prevent a completed group from entering training."""

    STALE = "stale"
    FULLY_MASKED = "fully_masked"
    PHYSICAL_GROUP_SIZE = "physical_group_size"
    BELOW_MINIMUM_GROUP_SIZE = "below_minimum_group_size"
    MISSING_ROLLOUT_LOGPROBS = "missing_rollout_logprobs"


@dataclass(frozen=True)
class AdmissionDecision:
    """Pure admission result; queue routing is owned by the caller."""

    rejections: tuple[AdmissionRejection, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.rejections

    @property
    def primary_rejection(self) -> AdmissionRejection | None:
        return self.rejections[0] if self.rejections else None


class GeneratedGroup(Protocol):
    trajectory_batch: Mapping[str, object]
    earliest_model_step: int


@dataclass(frozen=True)
class _BatchGroup:
    trajectory_batch: Mapping[str, object]
    earliest_model_step: int = 0


def _aligned_sequence(batch: Mapping[str, object], key: str, row_count: int) -> Sequence[object] | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence when present")
    if len(value) != row_count:
        raise ValueError(f"{key} must have one entry per response row, got {len(value)} and {row_count}")
    return value


@dataclass(frozen=True)
class _GroupFacts:
    physical_count: int
    trainable_count: int
    baseline_contributor_count: int
    has_rollout_logprobs: bool


def _inspect_group(group: GeneratedGroup) -> _GroupFacts:
    batch = group.trajectory_batch
    response_ids_value = batch.get("response_ids")
    if not isinstance(response_ids_value, Sequence) or isinstance(response_ids_value, (str, bytes)):
        raise ValueError("response_ids must be a sequence")
    if len(response_ids_value) == 0:
        raise ValueError("response_ids must contain at least one response row")
    response_ids = response_ids_value
    row_count = len(response_ids)
    loss_masks = _aligned_sequence(batch, "loss_masks", row_count)
    assert loss_masks is not None

    for row_index, (response, loss_mask) in enumerate(zip(response_ids, loss_masks, strict=True)):
        if not isinstance(response, Sequence) or not isinstance(loss_mask, Sequence):
            raise ValueError(f"response_ids and loss_masks row {row_index} must be sequences")
        if len(response) != len(loss_mask):
            raise ValueError(
                f"loss_masks row {row_index} must align with response_ids, got {len(loss_mask)} and {len(response)}"
            )

    is_last_step = _aligned_sequence(batch, "is_last_step", row_count)
    final_indices = (
        list(range(row_count)) if is_last_step is None else [i for i, value in enumerate(is_last_step) if value]
    )
    if not final_indices:
        raise ValueError("is_last_step must identify at least one final trial row")

    exclusions = _aligned_sequence(batch, "exclude_from_baseline", row_count)
    baseline_contributor_count = (
        len(final_indices) if exclusions is None else sum(not bool(exclusions[index]) for index in final_indices)
    )

    rollout_logprobs = _aligned_sequence(batch, "rollout_logprobs", row_count)
    has_trainable_rollout_logprobs = rollout_logprobs is not None
    if rollout_logprobs is not None:
        for row_index, (response, loss_mask, logprobs) in enumerate(
            zip(response_ids, loss_masks, rollout_logprobs, strict=True)
        ):
            if not isinstance(logprobs, Sequence) or len(response) != len(logprobs):
                raise ValueError(
                    f"rollout_logprobs row {row_index} must align with response_ids, "
                    f"got {len(logprobs) if isinstance(logprobs, Sequence) else 'non-sequence'} and {len(response)}"
                )
            if any(bool(mask) and logprob is None for mask, logprob in zip(loss_mask, logprobs, strict=True)):
                has_trainable_rollout_logprobs = False

    return _GroupFacts(
        physical_count=len(final_indices),
        trainable_count=sum(any(bool(token) for token in loss_mask) for loss_mask in loss_masks),
        baseline_contributor_count=baseline_contributor_count,
        has_rollout_logprobs=has_trainable_rollout_logprobs,
    )


class GroupAdmissionPolicy:
    """Evaluate completed groups without mutating async lifecycle state."""

    def __init__(
        self,
        invariant: GroupAdvantageInvariant,
        *,
        max_staleness_steps: int,
        rollout_logprobs_required: bool,
    ) -> None:
        self.invariant = invariant
        self.max_staleness_steps = max_staleness_steps
        self.rollout_logprobs_required = rollout_logprobs_required

    def is_stale(self, group: GeneratedGroup, *, global_step: int) -> bool:
        """Return whether the group's oldest sample exceeds the run's staleness cap."""
        return global_step - group.earliest_model_step > self.max_staleness_steps

    def evaluate(self, group: GeneratedGroup, *, global_step: int) -> AdmissionDecision:
        facts = _inspect_group(group)
        rejections = []
        if self.is_stale(group, global_step=global_step):
            rejections.append(AdmissionRejection.STALE)
        if facts.trainable_count == 0:
            rejections.append(AdmissionRejection.FULLY_MASKED)
        if (
            self.invariant.kind is not GroupAdvantageKind.NONE
            and facts.physical_count != self.invariant.physical_group_size
        ):
            rejections.append(AdmissionRejection.PHYSICAL_GROUP_SIZE)
        if (
            self.invariant.kind is GroupAdvantageKind.MINIMUM_BASELINE_ELIGIBLE
            and facts.baseline_contributor_count < self.invariant.minimum_group_size
        ):
            rejections.append(AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE)
        if self.rollout_logprobs_required and facts.trainable_count > 0 and not facts.has_rollout_logprobs:
            rejections.append(AdmissionRejection.MISSING_ROLLOUT_LOGPROBS)
        return AdmissionDecision(tuple(rejections))


def resolve_group_advantage_invariant(
    *, advantage_estimator: str, physical_group_size: int, minimum_group_size: int | None
) -> GroupAdvantageInvariant:
    """Resolve registry metadata and user input into one validated run invariant."""
    # Local import breaks the package cycle through skyrl_train.utils.__init__.
    from skyrl_train.utils.algorithm_registry import (  # noqa: PLC0415
        AdvantageEstimatorRegistry,
        ExactPhysicalGroup,
        MinimumBaselineEligibleGroup,
        NoGroupAdvantage,
    )

    contract = AdvantageEstimatorRegistry.group_contract(advantage_estimator)
    if isinstance(contract, MinimumBaselineEligibleGroup):
        if isinstance(minimum_group_size, bool) or not isinstance(minimum_group_size, int):
            raise ValueError(f"advantage estimator '{advantage_estimator}' requires integer group_advantage_min_size")
        return GroupAdvantageInvariant.minimum_baseline_eligible(
            physical_group_size=physical_group_size,
            minimum_group_size=minimum_group_size,
        )
    if minimum_group_size is not None:
        raise ValueError(
            f"advantage estimator '{advantage_estimator}' does not use group_advantage_min_size; set it to null"
        )
    if isinstance(contract, ExactPhysicalGroup):
        return GroupAdvantageInvariant.exact_physical(physical_group_size=physical_group_size)
    if isinstance(contract, NoGroupAdvantage):
        return GroupAdvantageInvariant.no_group_advantage(physical_group_size=physical_group_size)
    raise TypeError(f"unsupported group advantage contract: {type(contract).__name__}")


def assert_training_groups_eligible(
    trajectory_batch: Mapping[str, object],
    uids: Sequence[str],
    invariant: GroupAdvantageInvariant,
) -> None:
    """Fail when a synchronous or async training batch violates its group contract."""
    response_ids = trajectory_batch.get("response_ids")
    if not isinstance(response_ids, Sequence) or isinstance(response_ids, (str, bytes)):
        raise ValueError("response_ids must be a sequence")
    if len(response_ids) != len(uids):
        raise ValueError(f"uids must have one entry per response row, got {len(uids)} and {len(response_ids)}")

    grouped_indices: dict[str, list[int]] = {}
    for row_index, uid in enumerate(uids):
        grouped_indices.setdefault(uid, []).append(row_index)

    policy = GroupAdmissionPolicy(invariant, max_staleness_steps=0, rollout_logprobs_required=False)
    aligned_keys = ("response_ids", "loss_masks", "exclude_from_baseline", "rollout_logprobs", "is_last_step")
    for uid, indices in grouped_indices.items():
        group_batch = {}
        for key in aligned_keys:
            value = _aligned_sequence(trajectory_batch, key, len(response_ids))
            if value is not None:
                group_batch[key] = [value[index] for index in indices]
        decision = policy.evaluate(_BatchGroup(group_batch), global_step=0)
        if not decision.accepted:
            reasons = ", ".join(rejection.value for rejection in decision.rejections)
            raise ValueError(f"training group {uid!r} violates the resolved group invariant: {reasons}")
