"""Admission policy and checkpointable state for continuous rollout groups."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import List, TypeVar

from loguru import logger

from skyrl_train.async_rollout_state import GeneratedOutputGroup, GenerationBufferState
from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionProgressWatchdog,
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdmissionStalledError,
    TrainingGroupInvariantError,
)
from skyrl_train.rollout_buffer import RolloutBuffer
from skyrl_train.rollout_worker import GenerationStalledError
from skyrl_train.telemetry import record_generated_work
from skyrl_train.trajectory_runners.base import TrajectoryBatch
from skyrl_train.trajectory_runners.trajectory_processing import get_outcome_rewards
from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward

_QueueItem = TypeVar("_QueueItem")


def _drain_queue(queue: asyncio.Queue[_QueueItem]) -> List[_QueueItem]:
    """Remove available items without yielding."""
    items = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return items
    return items


@dataclass
class _GenerationQueues:
    rollout_buffer: RolloutBuffer[GeneratedOutputGroup]
    retries: asyncio.Queue[List[dict]]
    condition: asyncio.Condition
    active_producers: int = 0
    admitted_groups: List[GeneratedOutputGroup] = field(default_factory=list)
    admitted_groups_consumed: bool = False

    async def mark_producer_finished(self) -> None:
        """Wake admission when a generation worker permanently exits."""
        async with self.condition:
            if self.active_producers <= 0:
                raise RuntimeError("generation producer accounting underflow")
            self.active_producers -= 1
            self.condition.notify_all()

    def record_admitted(self, groups: List[GeneratedOutputGroup]) -> None:
        """Retain newly admitted groups until the step crosses its checkpoint boundary."""
        if self.admitted_groups_consumed:
            raise RuntimeError("cannot admit another group before clearing the consumed batch")
        self.admitted_groups.extend(groups)

    def mark_admitted_consumed(self) -> None:
        """Keep the trained batch available only to a final previous-checkpoint flush."""
        if not self.admitted_groups:
            raise RuntimeError("cannot consume an empty admitted batch")
        self.admitted_groups_consumed = True

    def clear_admitted(self) -> None:
        """Release the prior step's admitted groups before assembling the next step."""
        self.admitted_groups.clear()
        self.admitted_groups_consumed = False

    def snapshot(self) -> GenerationBufferState:
        """Copy queued and admitted work without yielding to another event-loop task."""
        admitted = [] if self.admitted_groups_consumed else list(self.admitted_groups)
        return self._snapshot(admitted)

    def shutdown_snapshot(self) -> GenerationBufferState:
        """Copy all work needed to recover from shutdown before the next checkpoint."""
        return self._snapshot(list(self.admitted_groups))

    def _snapshot(self, admitted_groups: List[GeneratedOutputGroup]) -> GenerationBufferState:
        retries = _drain_queue(self.retries)
        for prompts in retries:
            self.retries.put_nowait(prompts)
        return GenerationBufferState(self.rollout_buffer.snapshot(), retries, admitted_groups)


@dataclass
class _AdmissionPartition:
    accepted_groups: List[GeneratedOutputGroup]
    rejected_groups: List[tuple[GeneratedOutputGroup, AdmissionDecision]]
    discarded_groups: List[tuple[GeneratedOutputGroup, AdmissionDecision]]


@dataclass
class _DynamicSamplingCandidateMetrics:
    group_count: int = 0
    trajectory_count: int = 0
    optimization_reward_sum: float = 0.0
    outcome_reward_sum: float = 0.0
    passed_group_count: int = 0
    samples_per_group: int | None = None

    def observe(self, batch: TrajectoryBatch) -> None:
        rewards = batch["rewards"]
        outcomes = get_outcome_rewards(batch)
        group_size = len(outcomes)
        if self.samples_per_group is None:
            self.samples_per_group = group_size
        elif self.samples_per_group != group_size:
            raise ValueError(
                "dynamic sampling candidates must have a consistent physical group size: "
                f"got {self.samples_per_group} and {group_size}"
            )
        self.group_count += 1
        self.trajectory_count += group_size
        self.optimization_reward_sum += sum(NormalizedReward.from_output(reward).total for reward in rewards)
        self.outcome_reward_sum += sum(outcomes)
        self.passed_group_count += int(any(reward > 0.0 for reward in outcomes))

    def merge(self, other: "_DynamicSamplingCandidateMetrics") -> None:
        if other.samples_per_group is not None:
            if self.samples_per_group is None:
                self.samples_per_group = other.samples_per_group
            elif self.samples_per_group != other.samples_per_group:
                raise ValueError(
                    "dynamic sampling candidates changed physical group size within a training step: "
                    f"got {self.samples_per_group} and {other.samples_per_group}"
                )
        self.group_count += other.group_count
        self.trajectory_count += other.trajectory_count
        self.optimization_reward_sum += other.optimization_reward_sum
        self.outcome_reward_sum += other.outcome_reward_sum
        self.passed_group_count += other.passed_group_count


@dataclass
class _CandidateSelection:
    admitted_groups: List[GeneratedOutputGroup]
    surplus_groups: List[GeneratedOutputGroup]
    discarded_reasons: collections.Counter[str]
    candidate_metrics: _DynamicSamplingCandidateMetrics


class AsynchronousRolloutBuffer:
    """Select training batches from storage while producers continue independently."""

    def __init__(
        self,
        queues: _GenerationQueues,
        *,
        mini_batch_size: int,
        admission_policy: GroupAdmissionPolicy,
        selection_policy: GroupSelectionPolicy,
        max_candidate_groups: int | None,
        max_sample_batches: int,
        current_step: Callable[[], int],
        consumed_uids: Callable[[], set[str]],
        step_time_history: collections.deque[float],
        stall_timeout: float | None,
        on_admitted: Callable[[List[GeneratedOutputGroup]], Awaitable[None]],
        on_discarded: Callable[[int], Awaitable[None]],
        retain: Callable[[GeneratedOutputGroup], Awaitable[None]],
        metrics: Callable[[], dict],
    ):
        self.queues = queues
        self.mini_batch_size = mini_batch_size
        self._group_admission_policy = admission_policy
        self._group_selection_policy = selection_policy
        self._dynamic_sampling_type = selection_policy.sampling_type
        self._dynamic_sampling_max_candidate_groups = max_candidate_groups
        self._dynamic_sampling_max_sample_batches = max_sample_batches
        self._current_step = current_step
        self._consumed_uids = consumed_uids
        self._step_time_history = step_time_history
        self.group_admission_stall_timeout = stall_timeout
        self._on_admitted = on_admitted
        self._on_discarded = on_discarded
        self._retain = retain
        self._metrics = metrics
        self._groups_rejected_since_step = 0
        self._rejection_reasons_since_step: collections.Counter[str] = collections.Counter()
        self._groups_inspected_since_step = 0

    @property
    def global_step(self) -> int:
        return self._current_step()

    @property
    def all_metrics(self) -> dict:
        return self._metrics()

    def _record_admission_scan(
        self,
        rejected_groups: List[tuple[GeneratedOutputGroup, AdmissionDecision]],
        *,
        inspected_count: int,
    ) -> None:
        self._groups_rejected_since_step += len(rejected_groups)
        for _, decision in rejected_groups:
            assert decision.primary_rejection is not None
            self._rejection_reasons_since_step[decision.primary_rejection.value] += 1
        self._groups_inspected_since_step += inspected_count

    def _partition_completed_groups(
        self, completed_groups: List[GeneratedOutputGroup], occupied_uids: set[str]
    ) -> _AdmissionPartition:
        """Evaluate completed work and select at most one representative per UID."""
        decisions = [
            self._group_admission_policy.evaluate(group, global_step=self.global_step) for group in completed_groups
        ]
        selected_index_by_uid: dict[str, int] = {}
        for index, (group, decision) in enumerate(zip(completed_groups, decisions, strict=True)):
            if group.uid in occupied_uids:
                continue
            selected_index = selected_index_by_uid.get(group.uid)
            if selected_index is None or (decision.accepted and not decisions[selected_index].accepted):
                selected_index_by_uid[group.uid] = index

        duplicate_decision = AdmissionDecision((AdmissionRejection.DUPLICATE_UID,))
        accepted_groups = []
        rejected_groups = []
        discarded_groups = []
        for index, (group, decision) in enumerate(zip(completed_groups, decisions, strict=True)):
            if group.uid in occupied_uids or selected_index_by_uid[group.uid] != index:
                discarded_groups.append((group, duplicate_decision))
            elif decision.accepted:
                accepted_groups.append(group)
            else:
                rejected_groups.append((group, decision))
        return _AdmissionPartition(
            accepted_groups=accepted_groups,
            rejected_groups=rejected_groups,
            discarded_groups=discarded_groups,
        )

    def _publish_admission_metrics(
        self,
        *,
        dynamic_candidate_metrics: _DynamicSamplingCandidateMetrics,
        dynamic_discarded_count: int,
    ) -> None:
        rejected = self._groups_rejected_since_step
        inspected = self._groups_inspected_since_step
        assert inspected > 0, "An admitted training batch requires at least one inspected completed group"
        reason_counts = self._rejection_reasons_since_step
        self._groups_rejected_since_step = 0
        self._rejection_reasons_since_step = collections.Counter()
        self._groups_inspected_since_step = 0
        metrics = {
            "async/rejected_count": rejected,
            "async/rejected_rate": rejected / inspected,
        }
        if self._dynamic_sampling_type is DynamicSamplingType.FILTER:
            candidate_count = dynamic_candidate_metrics.group_count
            trajectory_count = dynamic_candidate_metrics.trajectory_count
            assert candidate_count > 0 and trajectory_count > 0
            metrics.update(
                {
                    "async/dynamic_sampling/candidate_count": candidate_count,
                    "async/dynamic_sampling/discarded_count": dynamic_discarded_count,
                    "async/dynamic_sampling/discarded_rate": (dynamic_discarded_count / candidate_count),
                    "async/dynamic_sampling/candidate_trajectory_count": trajectory_count,
                    "async/dynamic_sampling/candidate_optimization_reward_mean": (
                        dynamic_candidate_metrics.optimization_reward_sum / trajectory_count
                    ),
                    "async/dynamic_sampling/candidate_outcome_reward_mean": (
                        dynamic_candidate_metrics.outcome_reward_sum / trajectory_count
                    ),
                }
            )
            assert dynamic_candidate_metrics.samples_per_group is not None
            metrics[f"async/dynamic_sampling/candidate_pass_at_{dynamic_candidate_metrics.samples_per_group}"] = (
                dynamic_candidate_metrics.passed_group_count / candidate_count
            )
        metrics.update(
            {f"async/rejected_count/{reason.value}": reason_counts[reason.value] for reason in AdmissionRejection}
        )
        self.all_metrics.update(metrics)
        if rejected:
            logger.warning(
                f"Rejected {rejected} completed groups before step {self.global_step}; "
                f"reasons={dict(reason_counts)}. Waiting produced a full "
                f"{self.mini_batch_size}-group replacement batch."
            )
        if dynamic_discarded_count:
            logger.info(
                f"Dynamic sampling discarded {dynamic_discarded_count} of {candidate_count} "
                f"candidate groups before step {self.global_step}."
            )

    def _raise_admission_stall(
        self,
        elapsed: float,
        rejection_counts: collections.Counter[str],
        *,
        active_producers: int,
    ) -> None:
        """Bound a step that has admitted no new groups, even if producer tasks remain alive."""
        raise GroupAdmissionStalledError(
            f"Generation stalled: no groups admitted for {elapsed:.0f}s; "
            f"active_producers={active_producers}, "
            f"rejected_completions={dict(rejection_counts)}"
        )

    def _select_dynamic_sampling_candidates(
        self,
        candidates: List[GeneratedOutputGroup],
        *,
        available_slots: int,
    ) -> _CandidateSelection:
        admitted_groups = []
        discarded_reasons: collections.Counter[str] = collections.Counter()
        candidate_metrics = _DynamicSamplingCandidateMetrics()

        for candidate_index, group in enumerate(candidates):
            if len(admitted_groups) >= available_slots:
                return _CandidateSelection(
                    admitted_groups=admitted_groups,
                    surplus_groups=candidates[candidate_index:],
                    discarded_reasons=discarded_reasons,
                    candidate_metrics=candidate_metrics,
                )

            selection_result = self._group_selection_policy.evaluate(group)
            if self._dynamic_sampling_type is DynamicSamplingType.FILTER:
                candidate_metrics.observe(group.trajectory_batch)
            if selection_result is GroupSelectionResult.KEEP:
                admitted_groups.append(group)
            else:
                discarded_reasons[selection_result.value] += 1

        return _CandidateSelection(
            admitted_groups=admitted_groups,
            surplus_groups=[],
            discarded_reasons=discarded_reasons,
            candidate_metrics=candidate_metrics,
        )

    async def next_batch(self) -> List[GeneratedOutputGroup]:
        """Discard or retry rejected groups and wait for a full admitted mini-batch.

        Raises:
            GroupAdmissionStalledError: Live producers make no admission progress before the shared deadline.
            GenerationStalledError: The finite source is exhausted before a complete batch is assembled.
            RuntimeError: Dynamic sampling exhausts its per-step candidate budget.
        """
        queues = self.queues
        if queues.admitted_groups_consumed:
            raise RuntimeError("cannot assemble a new batch before clearing the previously consumed batch")
        accepted_groups = queues.admitted_groups
        loop = asyncio.get_event_loop()
        last_admitted_progress = loop.time()
        watchdog = AdmissionProgressWatchdog.start(
            now=last_admitted_progress,
            recent_step_times=self._step_time_history,
            timeout_override=self.group_admission_stall_timeout,
        )
        rejection_counts_since_admission: collections.Counter[str] = collections.Counter()
        dynamic_candidate_metrics = _DynamicSamplingCandidateMetrics()
        dynamic_discarded_count = 0
        await self._on_admitted(accepted_groups)

        while True:
            async with queues.condition:
                while len(accepted_groups) < self.mini_batch_size and queues.rollout_buffer.empty():
                    if queues.active_producers == 0:
                        raise GenerationStalledError(
                            "Generation exhausted its dataset before assembling a complete training batch: "
                            f"admitted={len(accepted_groups)}/{self.mini_batch_size}, "
                            f"dynamic_candidates={dynamic_candidate_metrics.group_count}, "
                            f"dynamic_discarded={dynamic_discarded_count}, "
                            f"rejections={dict(rejection_counts_since_admission)}"
                        )
                    now = loop.time()
                    elapsed = watchdog.elapsed(now=now)
                    remaining = watchdog.remaining(now=now)
                    if remaining <= 0:
                        self._raise_admission_stall(
                            elapsed,
                            rejection_counts_since_admission,
                            active_producers=queues.active_producers,
                        )
                    try:
                        await asyncio.wait_for(queues.condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        self._raise_admission_stall(
                            watchdog.elapsed(now=loop.time()),
                            rejection_counts_since_admission,
                            active_producers=queues.active_producers,
                        )

                completed_groups = await queues.rollout_buffer.next_batch(self.mini_batch_size)
                for group in completed_groups:
                    await self._retain(group)
                    batch = group.trajectory_batch
                    record_generated_work(batch["response_ids"], batch.get("is_last_step"), group.earliest_model_step)
                partition = self._partition_completed_groups(
                    completed_groups,
                    occupied_uids={group.uid for group in accepted_groups} | self._consumed_uids(),
                )
                for group, decision in partition.rejected_groups:
                    if decision.action is AdmissionAction.RETRY_PROMPT:
                        queues.retries.put_nowait(group.source_prompts)
                    elif decision.action is AdmissionAction.FAIL:
                        raise TrainingGroupInvariantError.from_generated_group(
                            uid=group.uid,
                            group=group,
                            decision=decision,
                            invariant=self._group_admission_policy.invariant,
                        )
                    assert decision.primary_rejection is not None
                    rejection_counts_since_admission[decision.primary_rejection.value] += 1

                selection = self._select_dynamic_sampling_candidates(
                    partition.accepted_groups,
                    available_slots=self.mini_batch_size - len(accepted_groups),
                )
                queues.record_admitted(selection.admitted_groups)
                dynamic_candidate_metrics.merge(selection.candidate_metrics)
                dynamic_discarded_this_scan = sum(selection.discarded_reasons.values())
                dynamic_discarded_count += dynamic_discarded_this_scan
                rejection_counts_since_admission.update(selection.discarded_reasons)

                for group in selection.surplus_groups:
                    queues.rollout_buffer.requeue(group)

                if selection.admitted_groups:
                    watchdog.observe(now=loop.time(), progressed=True)
                    rejection_counts_since_admission.clear()

                if len(accepted_groups) >= self.mini_batch_size:
                    batch = accepted_groups[: self.mini_batch_size]
                else:
                    batch = None
                queues.condition.notify_all()

            await self._on_admitted(selection.admitted_groups)

            self._record_admission_scan(
                partition.rejected_groups + partition.discarded_groups,
                inspected_count=len(completed_groups),
            )
            discarded_count = (
                len(partition.rejected_groups) + len(partition.discarded_groups) + dynamic_discarded_this_scan
            )
            if discarded_count:
                await self._on_discarded(discarded_count)

            if (
                batch is None
                and self._dynamic_sampling_max_candidate_groups is not None
                and dynamic_candidate_metrics.group_count >= self._dynamic_sampling_max_candidate_groups
            ):
                raise RuntimeError(
                    "Exiting training loop due to hitting dynamic sampling limit for filter strategy with "
                    f"{self._dynamic_sampling_max_sample_batches} max sample batches. "
                    f"Collected {len(accepted_groups)} of {self.mini_batch_size} required groups."
                )

            if batch is not None:
                break

        self._publish_admission_metrics(
            dynamic_candidate_metrics=dynamic_candidate_metrics,
            dynamic_discarded_count=dynamic_discarded_count,
        )
        return batch
