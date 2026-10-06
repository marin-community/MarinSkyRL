from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import ray

from skyrl_train.batch_metrics import consumed_work
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.telemetry import ConsumedWork, critical_phase
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import Timer


@dataclass
class BatchDiagnostics:
    metrics: dict[str, float]
    timings: dict[str, float]
    callback_rewards: tuple[float, ...] | None = None


@dataclass(frozen=True)
class BatchSummary:
    work: ConsumedWork
    group_count: int
    stop_metrics: dict[str, float]
    callback_rewards: tuple[float, ...] | None
    uids: tuple[str, ...]
    rollout_staleness: tuple[int, ...]
    response_lengths: tuple[int, ...]
    group_staleness: tuple[int, ...]


@dataclass(frozen=True)
class PolicyUpdate:
    submit_policy: Callable[[], list[ray.ObjectRef]]
    critic_batch: TrainingInputBatch | None


@dataclass(frozen=True)
class DriverBatchOperations:
    on_admitted: Callable[[list[RolloutGroup]], Awaitable[None]]
    build: Callable[
        [list[RolloutGroup], int, int, StepWallTime | None, BatchDiagnostics], Awaitable[TrainingInputBatch]
    ]
    forward: Callable[[TrainingInputBatch, BatchDiagnostics], Awaitable[TrainingInputBatch]]
    finalize: Callable[[TrainingInputBatch, StepWallTime | None, BatchDiagnostics], Awaitable[TrainingInputBatch]]
    dump: Callable[[TrainingInputBatch, StepWallTime | None, BatchDiagnostics], Awaitable[None]]
    submit_policy: Callable[[TrainingInputBatch], list[ray.ObjectRef]]
    optimize: Callable[[PolicyUpdate, BatchDiagnostics], Awaitable[dict]]


class BatchSource(Protocol):
    """Own one batch from admission through finalized input and optimizer submission."""

    async def admit(self, *, stall_timeout: float, diagnostics: BatchDiagnostics) -> dict[str, float]: ...
    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None: ...
    async def forward(self, *, step_wall: StepWallTime | None) -> None: ...
    async def finalize(self, *, step_wall: StepWallTime | None) -> None: ...
    def summary(self) -> BatchSummary: ...
    async def train(self, *, step_wall: StepWallTime | None) -> dict: ...
    async def release(self, error: BaseException | None = None) -> None: ...


class DriverBatchSource:
    """Run admitted groups through the trainer's configured driver operations."""

    def __init__(self, context: TrainingContext, operations: DriverBatchOperations):
        self.context = context
        self.operations = operations
        self._batch: TrainingInputBatch | None = None
        self._groups: list[RolloutGroup] = []
        self.diagnostics: BatchDiagnostics | None = None

    @property
    def batch(self) -> TrainingInputBatch:
        if self._batch is None:
            raise RuntimeError("driver batch is not prepared")
        return self._batch

    async def admit(self, *, stall_timeout: float, diagnostics: BatchDiagnostics) -> dict[str, float]:
        self.diagnostics = diagnostics
        self._groups, metrics = await self.context.next_batch(
            stall_timeout=stall_timeout, on_admitted=self.operations.on_admitted
        )
        self.batch_id = self.context.batch_id
        return metrics

    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None:
        await self.prepare_from_groups(
            self._groups,
            batch_id=self.batch_id,
            global_step=global_step,
            step_wall=step_wall,
            diagnostics=self.diagnostics,
        )

    async def prepare_from_groups(
        self,
        groups: list[RolloutGroup],
        *,
        batch_id: int,
        global_step: int,
        step_wall: StepWallTime | None,
        diagnostics: BatchDiagnostics,
    ) -> None:
        self._groups = groups
        self.batch_id = batch_id
        self.global_step = global_step
        batch = await self.operations.build(groups, batch_id, global_step, step_wall, diagnostics)
        await self.prepare_from_batch(batch, diagnostics=diagnostics)
        self._group_staleness = tuple(global_step - group.policy_step for group in groups)
        self._group_count = len(groups)

    async def prepare_from_batch(self, data: TrainingInputBatch, *, diagnostics: BatchDiagnostics) -> None:
        self._batch = data
        self.diagnostics = diagnostics
        self.global_step = data.metadata["global_step"]
        rows = data.batch_size - data.metadata.get("pad_size", 0)
        self._uids = tuple(data.metadata["uids"][:rows])
        self._staleness = tuple(data["rollout_staleness"][:rows].tolist())
        self._lengths = tuple(data["response_mask"][:rows].sum(-1).tolist())
        self._group_count = len(set(self._uids))
        self._group_staleness = tuple(dict(zip(self._uids, self._staleness)).values())
        self._empty = "ftpo_chosen_mask" in data and not data["loss_mask"].any()

    async def forward(self, *, step_wall: StepWallTime | None) -> None:
        if not self._empty:
            self._batch = await self.operations.forward(self.batch, self.diagnostics)

    async def finalize(self, *, step_wall: StepWallTime | None) -> None:
        if not self._empty:
            self._batch = await self.operations.finalize(self.batch, step_wall, self.diagnostics)

    async def dump(self, *, step_wall: StepWallTime | None) -> None:
        await self.operations.dump(self.batch, step_wall, self.diagnostics)

    def summary(self) -> BatchSummary:
        return BatchSummary(
            consumed_work(self.batch),
            self._group_count,
            self.batch.metadata["consumed_stop_metrics"],
            self.diagnostics.callback_rewards,
            self._uids,
            self._staleness,
            self._lengths,
            self._group_staleness,
        )

    async def train(self, *, step_wall: StepWallTime | None) -> dict:
        if self._empty:
            self.diagnostics.timings["train_critic_and_policy"] = 0.0
            self.diagnostics.metrics.update({"policy/policy_update_steps": 0.0, "ftpo/empty_batch": 1.0})
            return {"policy_update_steps": 0.0}
        await self.dump(step_wall=step_wall)
        if step_wall is not None:
            step_wall.start("policy_training")
        with Timer("train_critic_and_policy", self.diagnostics.timings), critical_phase("train_step", self.global_step):
            return await self.operations.optimize(
                PolicyUpdate(lambda: self.operations.submit_policy(self.batch), self.batch), self.diagnostics
            )

    async def release(self, error: BaseException | None = None) -> None:
        self._batch = None
        self._groups = []
        self.diagnostics = None
