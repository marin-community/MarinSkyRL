"""Shared state records for fully asynchronous rollout generation."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import List, Protocol

import torch

from skyrl_train.group_admission import sampled_token_version_bounds
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.rollout_observability import consumed_stop_metrics
from skyrl_train.trajectory_runners.base import TrajectoryBatch


@dataclass
class GeneratedOutputGroup:
    """One prompt's rollout samples and the metadata needed to retry them."""

    trajectory_batch: TrajectoryBatch
    uid: str
    earliest_model_step: int
    source_prompts: List[dict]
    # Process-local observations deliberately omitted from checkpoint serialization.
    completed_at: float | None = None
    telemetry_attempt_id: str | None = None
    admitted_at: float | None = None
    telemetry_finished: bool = False
    latest_model_step: int | None = None

    def __post_init__(self) -> None:
        if self.latest_model_step is None:
            self.latest_model_step = self.earliest_model_step
        if self.latest_model_step < self.earliest_model_step:
            raise ValueError("latest model step cannot precede the earliest model step")


def consumed_token_version_metrics(groups: List[GeneratedOutputGroup], step: int) -> dict[str, float]:
    """Summarize loss-token ages with explicit coverage for legacy or unknown evidence."""
    if not any(
        group.trajectory_batch.get("rollout_versions") is not None
        or group.trajectory_batch.get("rollout_abort_counts") is not None
        for group in groups
    ):
        return {}
    ages = []
    unknown = 0
    spreads = []
    newest_ages = []
    aborts = []
    response_count = 0
    for group in groups:
        batch = group.trajectory_batch
        rows = batch.get("rollout_versions") or [[None] * len(ids) for ids in batch["response_ids"]]
        bounds = sampled_token_version_bounds(batch)
        if bounds is not None:
            spreads.append(bounds[1] - bounds[0])
            newest_ages.append(step - bounds[1])
        response_count += len(rows)
        aborts.extend(value for value in (batch.get("rollout_abort_counts") or []) if value is not None)
        for row, mask in zip(rows, batch["loss_masks"], strict=True):
            for version, sampled in zip(row, mask, strict=True):
                if not sampled:
                    continue
                if version is None:
                    unknown += 1
                else:
                    age = step - version
                    if age < 0:
                        raise ValueError("Consumed token version is newer than the consuming model step")
                    ages.append(age)
    total = len(ages) + unknown
    metrics = {
        "async/token_version_known_tokens": len(ages),
        "async/token_version_unknown_tokens": unknown,
        "async/token_version_known_fraction": len(ages) / total if total else 0.0,
        "async/token_version_complete_group_fraction": len(spreads) / len(groups),
        "async/response_abort_count_known_fraction": len(aborts) / response_count if response_count else 0.0,
    }
    if aborts:
        metrics["async/response_abort_count"] = sum(aborts)
    if spreads:
        metrics.update(
            {
                "async/version_spread_mean": sum(spreads) / len(spreads),
                "async/version_spread_max": max(spreads),
                "async/staleness_newest_mean": sum(newest_ages) / len(newest_ages),
            }
        )
    if ages:
        for bucket in (0, 1, 2):
            metrics[f"async/token_age_frac/{bucket}"] = sum(age == bucket for age in ages) / len(ages)
        metrics["async/token_age_frac/3plus"] = sum(age >= 3 for age in ages) / len(ages)
    return metrics


@dataclass(frozen=True)
class PreparedAsyncCohort:
    """One admitted cohort, prepared once, with an optimizer-update resume cursor.

    Tensors in ``batch`` belong to this record. Partitions copy tensors and metadata
    so worker preparation cannot change the old log probabilities or advantages
    retained for a later update or a previous-checkpoint shutdown flush.
    """

    batch: TrainingInputBatch
    groups: List[GeneratedOutputGroup]
    admission_step: int
    dp_size: int
    mini_batch_groups: int
    samples_per_prompt: int
    next_update: int = 0

    def __post_init__(self) -> None:
        if min(self.dp_size, self.mini_batch_groups, self.samples_per_prompt, self.admission_step) < 1:
            raise ValueError("prepared cohort geometry and admission step must be positive")
        if len(self.groups) % self.mini_batch_groups or self.mini_batch_groups % self.dp_size:
            raise ValueError("prepared cohort must contain complete DP-striped optimizer minibatches")
        if not self.groups or not 0 <= self.next_update <= self.num_updates:
            raise ValueError("invalid prepared cohort update cursor")
        uids = [group.uid for group in self.groups]
        if len(set(uids)) != len(uids):
            raise ValueError("prepared cohort group UIDs must be unique")
        expected = [uid for uid in uids for _ in range(self.samples_per_prompt)]
        if len(self.batch) != len(expected) or self.batch.metadata.get("uids") != expected:
            raise ValueError("prepared cohort rows must match complete, unpadded source groups")
        if any(group.earliest_model_step > self.admission_step for group in self.groups):
            raise ValueError("prepared cohort cannot contain future admission stamps")
        if any(self.batch.get(key) is None for key in ("action_log_probs", "advantages")):
            raise ValueError("prepared cohort requires frozen old log probabilities and advantages")

    @property
    def num_updates(self) -> int:
        return len(self.groups) // self.mini_batch_groups

    def group_indices(self, update: int | None = None) -> list[int]:
        """Select the same local strips as MeshDispatch then TrainingBatchIterator."""
        update = self.next_update if update is None else update
        if not 0 <= update < self.num_updates:
            raise ValueError("prepared cohort has no such remaining optimizer update")
        rank_groups = len(self.groups) // self.dp_size
        rank_mini = self.mini_batch_groups // self.dp_size
        return [
            rank * rank_groups + update * rank_mini + offset
            for rank in range(self.dp_size)
            for offset in range(rank_mini)
        ]

    def pending_groups(self) -> list[GeneratedOutputGroup]:
        return [
            self.groups[index]
            for update in range(self.next_update, self.num_updates)
            for index in self.group_indices(update)
        ]

    def partition(self, consume_step: int) -> TrainingInputBatch:
        """Return one update's frozen inputs with its unclipped consume-time ages."""
        if consume_step != self.admission_step + self.next_update:
            raise ValueError("prepared cohort cursor must agree with the successful-update clock")
        group_indices = self.group_indices()
        rows = [
            index * self.samples_per_prompt + sample
            for index in group_indices
            for sample in range(self.samples_per_prompt)
        ]
        part = TrainingInputBatch(
            {key: value[rows].clone() if value is not None else None for key, value in self.batch.items()}
        )
        part.metadata = deepcopy(self.batch.metadata)
        for key in ("uids", "exclude_from_baseline", "trajectory_ids"):
            if key in part.metadata:
                value = part.metadata[key]
                part.metadata[key] = [value[row] for row in rows]
        ages = [consume_step - self.groups[index].earliest_model_step for index in group_indices]
        part["rollout_age"] = torch.tensor(ages, dtype=torch.int32, device=part["loss_mask"].device).repeat_interleave(
            self.samples_per_prompt
        )
        part.metadata.update(
            async_cohort_admission_step=self.admission_step,
            async_cohort_update_index=self.next_update,
            async_cohort_admission_ages=[age - self.next_update for age in ages],
        )
        if "consumed_stop_metrics" in part.metadata:
            reasons = [
                reason
                for index in group_indices
                for reason in (
                    self.groups[index].trajectory_batch.get("stop_reasons") or [None] * self.samples_per_prompt
                )
            ]
            part.metadata["consumed_stop_metrics"] = consumed_stop_metrics(reasons, len(rows))
        part.metadata["avg_response_length"] = float(part["response_mask"].sum().item()) / len(rows)
        rewards = part.pop("prepared_rewards", None)
        if rewards is not None:
            advantages = part["advantages"].masked_select(part["response_mask"].bool())
            part.metadata["metrics"] = {
                "avg_final_rewards": rewards.sum(-1).float().mean().item(),
                "avg_response_length": part.metadata["avg_response_length"],
                "avg_advantages": advantages.mean().item(),
                "avg_advantages_abs": advantages.abs().mean().item(),
            }
        return part

    def advanced(self) -> "PreparedAsyncCohort":
        if self.next_update == self.num_updates:
            raise ValueError("prepared cohort is already consumed")
        return replace(self, next_update=self.next_update + 1)


@dataclass
class GenerationBufferState:
    """Completed, admitted, and retryable rollout work stored with a checkpoint."""

    completed_groups: List[GeneratedOutputGroup]
    retry_prompts: List[List[dict]]
    admitted_groups: List[GeneratedOutputGroup] = field(default_factory=list)
    prepared_cohort: PreparedAsyncCohort | None = None

    def pending_uids(self) -> set[str]:
        """Return dataset UIDs whose work survives in this checkpoint."""
        uids = set()
        for group in self.completed_groups:
            if not isinstance(group.uid, str):
                raise ValueError("completed generation group uid must be a string")
            uids.add(group.uid)
        for group in self.admitted_groups:
            if not isinstance(group.uid, str):
                raise ValueError("admitted generation group uid must be a string")
            uids.add(group.uid)
        if self.prepared_cohort is not None:
            uids.update(group.uid for group in self.prepared_cohort.pending_groups())
        for prompts in self.retry_prompts:
            for prompt in prompts:
                uid = prompt.get("uid")
                if not isinstance(uid, str):
                    raise ValueError("retry prompt uid must be a string")
                uids.add(uid)
        return uids


class GenerationQueuesProvider(Protocol):
    """Live generation queues that can provide checkpoint state."""

    def snapshot(self) -> GenerationBufferState: ...

    def shutdown_snapshot(self) -> GenerationBufferState: ...
