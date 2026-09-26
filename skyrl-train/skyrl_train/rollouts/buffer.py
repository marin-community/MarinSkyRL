"""Rollout buffer: leases generation capacity to rollout workers and selects training batches.

A rollout worker writes each completed prompt group through a ``RolloutWriter``. The writer checks the
group's content where the payload already is and commits a small verdict together with a reference to the
payload, so the buffer selects batches without reading payloads. In memory mode ``RolloutBuffer`` runs as a
Ray actor and payloads stay in Ray's object store, owned by that actor so they outlive the worker that
wrote them. The trainer fetches only the payloads it trains on.
"""

from __future__ import annotations

import asyncio
import collections
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import ray
from ray.actor import ActorHandle

from skyrl_train.dynamic_sampling import DynamicSamplingType, GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import (
    AdmissionRejection,
    GroupAdmissionPolicy,
    GroupAdmissionStalledError,
    TrainingGroupInvariantError,
)
from skyrl_train.rollouts.loader import JudgedGroup
from skyrl_train.telemetry import GeneratedWork
from skyrl_train.trajectory_runners.trajectory_processing import get_outcome_rewards
from skyrl_train.trajectory_runners.trajectory_reward_shaping import NormalizedReward
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch


@dataclass(frozen=True)
class RolloutBufferConfig:
    """Batch shape, generation concurrency, and selection rules of one training run.

    ``max_in_flight`` caps concurrent rollouts and ``max_candidate_groups`` bounds the dynamic-sampling candidates
    one batch may inspect; None leaves either unbounded.
    """

    batch_size: int
    max_in_flight: int | None
    max_staleness_steps: int
    dynamic_sampling: DynamicSamplingType | None
    max_candidate_groups: int | None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"rollout batch size must be positive, got {self.batch_size}")
        if self.max_in_flight is not None and self.max_in_flight < 1:
            raise ValueError(f"max in-flight rollouts must be positive, got {self.max_in_flight}")
        if self.max_staleness_steps < 0:
            raise ValueError(f"max_staleness_steps must be non-negative, got {self.max_staleness_steps}")

    @property
    def max_untrained_groups(self) -> int:
        """Groups leased, committed, or batched that the next ``max_staleness_steps + 1`` steps can train on."""
        return (self.max_staleness_steps + 1) * self.batch_size


@dataclass(frozen=True)
class RolloutLease:
    """Permission to generate one prompt group with the policy published at ``policy_step``."""

    lease_id: str
    policy_step: int


@dataclass(frozen=True)
class RolloutTask:
    """One prompt group to generate under a buffer lease."""

    lease: RolloutLease
    prompt: dict
    request: TrajectoryRequestBatch


@dataclass
class RolloutGroup:
    """One prompt's samples, generated under one lease."""

    trajectory_batch: TrajectoryBatch
    uid: str
    policy_step: int
    prompt: dict
    request: TrajectoryRequestBatch


@dataclass(frozen=True)
class GroupRewards:
    """Per-sample rewards of one group: each sample's optimization reward total and its outcome reward."""

    optimization: tuple[float, ...]
    outcome: tuple[float, ...]

    @classmethod
    def from_batch(cls, batch: TrajectoryBatch) -> GroupRewards:
        return cls(
            optimization=tuple(NormalizedReward.from_output(reward).total for reward in batch["rewards"]),
            outcome=tuple(get_outcome_rewards(batch)),
        )


@dataclass(frozen=True)
class RolloutVerdict:
    """What the buffer needs to know about a group's content to select it.

    ``selection`` and ``rewards`` are None when a content rejection already excludes the group.
    """

    uid: str
    rejections: tuple[AdmissionRejection, ...]
    selection: GroupSelectionResult | None
    rewards: GroupRewards | None
    work: GeneratedWork

    @property
    def trainable(self) -> bool:
        return not self.rejections and self.selection is GroupSelectionResult.KEEP


@dataclass(frozen=True)
class RolloutContentPolicy:
    """Content checks that do not depend on when a group is read."""

    admission: GroupAdmissionPolicy
    selection: GroupSelectionPolicy

    def verdict(self, group: RolloutGroup) -> RolloutVerdict:
        """Judge one group, raising when it violates the run's structural contract."""
        decision = self.admission.evaluate(group)
        if decision.fatal:
            raise TrainingGroupInvariantError.from_generated_group(
                uid=group.uid, group=group, decision=decision, invariant=self.admission.invariant
            )
        batch = group.trajectory_batch
        work = GeneratedWork.from_batch(batch["response_ids"], batch.get("is_last_step"))
        if not decision.accepted:
            return RolloutVerdict(group.uid, decision.rejections, None, None, work)
        return RolloutVerdict(group.uid, (), self.selection.evaluate(group), GroupRewards.from_batch(batch), work)


class RolloutWriter(Protocol):
    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None: ...


@dataclass(frozen=True)
class MemoryRolloutWriter:
    """Store payloads in Ray's object store, owned by the buffer actor, and commit their verdicts."""

    buffer: ActorHandle
    content_policy: RolloutContentPolicy

    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None:
        verdict = self.content_policy.verdict(group)
        payload = []
        if verdict.trainable:
            payload.append(await asyncio.to_thread(ray.put, group, _owner=self.buffer))
        # Nested in a list so Ray passes the reference instead of resolving it.
        await self.buffer.commit.remote(lease.lease_id, group.prompt, verdict, payload)


@dataclass
class ReadyRollout:
    """A committed group that no batch has taken yet.

    ``payload`` holds a reference to the ``RolloutGroup``; it is empty when the verdict excludes the group.
    ``committed_at`` is the buffer process's monotonic time at commit, and None for a group restored from a
    checkpoint.
    """

    lease_id: str
    policy_step: int
    prompt: dict
    verdict: RolloutVerdict
    payload: list
    committed_at: float | None


@dataclass(frozen=True)
class GroupDisposition:
    """How one committed group left the buffer: ``consumed`` by a batch or an admission or selection outcome."""

    disposition: str
    tokens: int
    dwell_seconds: float | None


@dataclass(frozen=True)
class BufferSnapshot:
    """Committed groups, prompts awaiting regeneration, and leases not yet committed, for a checkpoint."""

    ready: list[ReadyRollout]
    retries: list[dict]
    leases: frozenset[str]


@dataclass(frozen=True)
class BatchSelection:
    """How a completed batch was selected: its metrics and every group judged for it, kept or discarded."""

    metrics: dict[str, float]
    judged: list[JudgedGroup]


@dataclass(frozen=True)
class Admission:
    """Progress toward the current batch since the previous ``admit`` call.

    ``selection`` is set only on the call that completes the batch.
    """

    payloads: list
    retries: list[dict]
    generated: list[tuple[int, GeneratedWork]]
    dispositions: list[GroupDisposition]
    ready_count: int
    selection: BatchSelection | None


@dataclass
class _SelectionStats:
    """Admission and dynamic-sampling outcomes since the previous batch."""

    inspected: int = 0
    rejections: collections.Counter[AdmissionRejection] = field(default_factory=collections.Counter)
    candidates: int = 0
    candidate_samples: int = 0
    samples_per_group: int | None = None
    optimization_reward_sum: float = 0.0
    outcome_reward_sum: float = 0.0
    passed: int = 0
    dynamic_discarded: int = 0
    judged: list[JudgedGroup] = field(default_factory=list)

    def reject(self, rejection: AdmissionRejection) -> None:
        self.inspected += 1
        self.rejections[rejection] += 1

    def observe_candidate(self, rewards: GroupRewards) -> None:
        sample_count = len(rewards.optimization)
        if self.samples_per_group is None:
            self.samples_per_group = sample_count
        elif self.samples_per_group != sample_count:
            raise ValueError(
                "dynamic sampling candidates must have a consistent physical group size: "
                f"got {self.samples_per_group} and {sample_count}"
            )
        self.candidates += 1
        self.candidate_samples += sample_count
        self.optimization_reward_sum += sum(rewards.optimization)
        self.outcome_reward_sum += sum(rewards.outcome)
        self.passed += int(any(reward > 0.0 for reward in rewards.outcome))

    def metrics(self, dynamic_sampling: DynamicSamplingType | None) -> dict[str, float]:
        rejected = sum(self.rejections.values())
        metrics = {
            "async/rejected_count": rejected,
            "async/rejected_rate": rejected / self.inspected,
            **{f"async/rejected_count/{reason.value}": self.rejections[reason] for reason in AdmissionRejection},
        }
        if dynamic_sampling is DynamicSamplingType.FILTER:
            metrics.update(
                {
                    "async/dynamic_sampling/candidate_count": self.candidates,
                    "async/dynamic_sampling/discarded_count": self.dynamic_discarded,
                    "async/dynamic_sampling/discarded_rate": self.dynamic_discarded / self.candidates,
                    "async/dynamic_sampling/candidate_trajectory_count": self.candidate_samples,
                    "async/dynamic_sampling/candidate_optimization_reward_mean": (
                        self.optimization_reward_sum / self.candidate_samples
                    ),
                    "async/dynamic_sampling/candidate_outcome_reward_mean": (
                        self.outcome_reward_sum / self.candidate_samples
                    ),
                    f"async/dynamic_sampling/candidate_pass_at_{self.samples_per_group}": self.passed / self.candidates,
                }
            )
        return metrics


class RolloutBuffer:
    """Lease accounting, committed groups, and batch selection for one training run.

    The trainer publishes each policy step after syncing its weights to inference. Leases open only after
    the first publish. Besides any ``max_in_flight`` cap, leases are bounded so that groups not yet trained (leased,
    committed, or in the current batch) never exceed ``max_staleness_steps + 1`` batches: a group leased now
    can only be trained within that many steps, so generating more would only produce stale work.

    Every commit is judged immediately: a group too stale for the current step returns its prompt for
    regeneration, a content rejection or duplicate UID is dropped, and dynamic sampling decides the rest.
    Groups that arrive after the batch is full wait for the next step.
    """

    def __init__(self, config: RolloutBufferConfig):
        self.config = config
        self._policy_step = 0
        self._leases: dict[str, int] = {}
        self._ready: collections.deque[ReadyRollout] = collections.deque()
        self._admitted: list[ReadyRollout] = []
        self._unreported: list[ReadyRollout] = []
        self._batch_taken = False
        self._retries: list[dict] = []
        self._generated: list[tuple[int, GeneratedWork]] = []
        self._dispositions: list[GroupDisposition] = []
        self._stats = _SelectionStats()
        self._changed = asyncio.Condition()

    def _capacity(self) -> int:
        if self._policy_step == 0:
            return 0
        running = len(self._leases)
        batched = self.config.batch_size if self._batch_taken else len(self._admitted)
        available = self.config.max_untrained_groups - (running + len(self._ready) + batched)
        if self.config.max_in_flight is None:
            return available
        return min(self.config.max_in_flight - running, available)

    async def acquire_lease(self) -> RolloutLease:
        """Wait for generation capacity and lease it at the current policy step."""
        async with self._changed:
            await self._changed.wait_for(lambda: self._capacity() > 0)
            lease = RolloutLease(uuid.uuid4().hex, self._policy_step)
            self._leases[lease.lease_id] = lease.policy_step
            return lease

    async def commit(self, lease_id: str, prompt: dict, verdict: RolloutVerdict, payload: list) -> None:
        """Record a written group and release its lease."""
        async with self._changed:
            policy_step = self._leases.pop(lease_id)
            self._generated.append((policy_step, verdict.work))
            self._ready.append(ReadyRollout(lease_id, policy_step, prompt, verdict, payload, time.monotonic()))
            self._select()
            self._changed.notify_all()

    async def publish(self, policy_step: int) -> None:
        """Acknowledge the taken batch, if any, and lease at the newly synced policy step."""
        async with self._changed:
            if policy_step <= self._policy_step:
                raise ValueError(f"policy step must advance past {self._policy_step}, got {policy_step}")
            if self._policy_step and not self._batch_taken:
                raise RuntimeError("cannot publish a new policy step before taking the current batch")
            self._policy_step = policy_step
            self._admitted = []
            self._batch_taken = False
            self._select()
            self._changed.notify_all()

    async def admit(self, timeout: float) -> Admission:
        """Wait up to ``timeout`` seconds for the current batch to progress.

        Returns newly admitted payloads and prompts to regenerate. The call that completes the batch also
        returns how the batch was selected; the batch then counts as taken until the next ``publish``.

        Raises:
            GroupAdmissionStalledError: Nothing was admitted or returned for regeneration within ``timeout``.
            RuntimeError: Dynamic sampling inspected its per-batch candidate budget without filling the batch.
        """
        async with self._changed:
            if self._batch_taken:
                raise RuntimeError("the current batch was already taken; publish the next policy step first")
            try:
                await asyncio.wait_for(
                    self._changed.wait_for(
                        lambda: self._unreported or self._retries or self._batch_complete() or self._over_budget()
                    ),
                    timeout,
                )
            except TimeoutError as error:
                raise GroupAdmissionStalledError(
                    f"no rollout group admitted for {timeout:.0f}s: policy_step={self._policy_step} "
                    f"admitted={len(self._admitted)}/{self.config.batch_size} leases={len(self._leases)} "
                    f"ready={len(self._ready)} rejections={dict(self._stats.rejections)}"
                ) from error
            selection = None
            if self._batch_complete():
                for rollout in self._admitted:
                    self._dispose(rollout, "consumed")
                selection = BatchSelection(self._stats.metrics(self.config.dynamic_sampling), self._stats.judged)
                self._stats = _SelectionStats()
                self._batch_taken = True
            elif self._over_budget():
                raise RuntimeError(
                    "dynamic sampling inspected its limit of "
                    f"{self.config.max_candidate_groups} candidate groups with "
                    f"{len(self._admitted)} of {self.config.batch_size} admitted"
                )
            admission = Admission(
                payloads=[ref for rollout in self._unreported for ref in rollout.payload],
                retries=self._retries,
                generated=self._generated,
                dispositions=self._dispositions,
                ready_count=len(self._ready),
                selection=selection,
            )
            self._unreported, self._retries, self._generated, self._dispositions = [], [], [], []
            self._changed.notify_all()
            return admission

    def snapshot(self) -> BufferSnapshot:
        """Copy committed groups not yet in a taken batch, and prompts awaiting regeneration."""
        admitted = [] if self._batch_taken else self._admitted
        return BufferSnapshot(
            ready=[*admitted, *self._ready], retries=list(self._retries), leases=frozenset(self._leases)
        )

    async def restore(self, snapshot: BufferSnapshot) -> None:
        """Load a checkpoint's groups into an unpublished, empty buffer."""
        async with self._changed:
            if self._policy_step or self._leases or self._ready or self._retries:
                raise RuntimeError("a rollout buffer can only be restored before training starts")
            self._ready.extend(snapshot.ready)
            self._retries.extend(snapshot.retries)

    def _reject(self, rollout: ReadyRollout, rejection: AdmissionRejection) -> None:
        self._stats.reject(rejection)
        self._dispose(rollout, rejection.value)

    def _dispose(self, rollout: ReadyRollout, disposition: str) -> None:
        dwell = None if rollout.committed_at is None else time.monotonic() - rollout.committed_at
        self._dispositions.append(GroupDisposition(disposition, rollout.verdict.work.generated_token_count, dwell))

    def _batch_complete(self) -> bool:
        return len(self._admitted) == self.config.batch_size and not self._batch_taken

    def _over_budget(self) -> bool:
        limit = self.config.max_candidate_groups
        return limit is not None and self._stats.candidates >= limit

    def _select(self) -> None:
        """Judge committed groups in arrival order against the current step's batch."""
        if self._policy_step == 0:
            return
        batch_uids = {rollout.verdict.uid for rollout in self._admitted}
        waiting: collections.deque[ReadyRollout] = collections.deque()
        for rollout in self._ready:
            verdict = rollout.verdict
            if self._policy_step - rollout.policy_step > self.config.max_staleness_steps:
                self._reject(rollout, AdmissionRejection.STALE)
                self._retries.append(rollout.prompt)
            elif verdict.rejections:
                self._reject(rollout, verdict.rejections[0])
            elif self._batch_taken or len(self._admitted) == self.config.batch_size:
                waiting.append(rollout)
            elif verdict.uid in batch_uids:
                self._reject(rollout, AdmissionRejection.DUPLICATE_UID)
            else:
                self._stats.inspected += 1
                self._stats.judged.append(JudgedGroup(verdict.uid, verdict.rewards.optimization))
                if self.config.dynamic_sampling is DynamicSamplingType.FILTER:
                    self._stats.observe_candidate(verdict.rewards)
                if verdict.selection is not GroupSelectionResult.KEEP:
                    self._stats.dynamic_discarded += 1
                    self._dispose(rollout, verdict.selection.value)
                    continue
                self._admitted.append(rollout)
                self._unreported.append(rollout)
                batch_uids.add(verdict.uid)
        self._ready = waiting
