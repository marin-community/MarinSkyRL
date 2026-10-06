import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import ray
import torch
from omegaconf import DictConfig

from skyrl_train.batch_assembly import BatchLoadResult, BatchPlan, fields_from_facts, outcome_advantages, plan_batch
from skyrl_train.batch_metrics import consumed_stop_metrics, consumed_work, group_reward_metrics
from skyrl_train.rollouts.buffer import RolloutGroup, RowFacts
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.telemetry import ConsumedWork, critical_phase
from skyrl_train.timing_observability import StepWallTime
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.rollout_metrics import (
    merge_rollout_observations,
    reward_metrics,
    rollout_metrics,
    staleness_metrics,
)
from skyrl_train.utils import Timer
from skyrl_train.workers.worker import PPORayActorGroup


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


class WorkerBatchSource:
    """Retain each DP slice on policy actors from admission through optimization."""

    def __init__(
        self,
        context: TrainingContext,
        policy: PPORayActorGroup,
        algorithm: DictConfig,
        *,
        replay: bool,
        num_experts: int | None,
        colocate_all: bool,
        n_samples_per_prompt: int,
        optimize: Callable[[PolicyUpdate, BatchDiagnostics], Awaitable[dict]],
    ):
        self.context = context
        self.policy = policy
        self.algorithm = algorithm
        self.replay = replay
        self.num_experts = num_experts
        self.colocate_all = colocate_all
        self.n_samples_per_prompt = n_samples_per_prompt
        self.optimize = optimize
        self._pending: list[ray.ObjectRef] = []
        self._references = []
        self.plan: BatchPlan | None = None
        self.diagnostics: BatchDiagnostics | None = None

    async def admit(self, *, stall_timeout: float, diagnostics: BatchDiagnostics) -> dict[str, float]:
        self.diagnostics = diagnostics
        self.metadata = await self.context.next_batch_metadata(stall_timeout=stall_timeout)
        self.reader = self.context.reader(self.metadata, stall_timeout=stall_timeout)
        return self.metadata.metrics

    async def prepare(self, *, global_step: int, step_wall: StepWallTime | None) -> None:
        if step_wall is not None:
            step_wall.start("batch_assembly")
        groups = self.metadata.groups
        facts = [group.row_facts for group in groups]
        if any(fact is None for fact in facts):
            raise ValueError("worker batches require rollout row facts; restore this checkpoint with the driver source")
        fields = fields_from_facts(facts)
        if not self.replay:
            fields = fields.without_routes()
        uids = tuple(group.uid for group in groups for _ in group.row_facts.response_len)
        staleness = tuple(global_step - group.policy_step for group in groups for _ in group.row_facts.response_len)
        self.plan = plan_batch(
            batch_id=self.metadata.batch_id,
            global_step=global_step,
            uids=uids,
            facts=RowFacts.concatenate(facts),
            fields=fields,
            dp_size=self.policy.actor_infos[0].rank.dp_size,
            rollout_staleness=staleness,
            num_experts=self.num_experts,
        )
        advantages = outcome_advantages(self.plan.facts, uids, self.algorithm)
        self._pending = self.policy.read_batch_groups(self.plan, self.reader)
        self._references = list(await self.context.wait(asyncio.gather(*self._pending)))
        self._pending = self.policy.load_batch(self.plan, self.reader, self._references, advantages)
        loaded = await self.context.wait(asyncio.gather(*self._pending))
        self._pending = []
        self._record_observations(loaded, global_step=global_step)

    def _record_observations(self, loaded: Sequence[BatchLoadResult], *, global_step: int) -> None:
        groups = self.metadata.groups
        uids = self.plan.uids
        observed = [(index, obs) for output in loaded for index, obs in output.observations]
        if sorted(index for index, _ in observed) != [group.index for group in groups]:
            raise ValueError("worker observations must cover each admitted group exactly once")
        observations = merge_rollout_observations([obs for _, obs in sorted(observed, key=lambda item: item[0])])
        self.diagnostics.metrics.update(
            rollout_metrics(observations, tis_lcs_alert_threshold=float(self.algorithm.tis_lcs_alert_threshold))
        )
        self.diagnostics.metrics.update(
            reward_metrics(observations, uids, n_samples_per_prompt=self.n_samples_per_prompt, step_wise=False)
        )
        self.diagnostics.metrics.update(
            staleness_metrics(
                [group.policy_step for group in groups],
                global_step=global_step,
                max_staleness_steps=self.context.config.max_staleness_steps,
            )
        )
        self.diagnostics.callback_rewards = observations.scalar_rewards
        self.diagnostics.metrics.update(
            group_reward_metrics(
                uids,
                torch.from_numpy(self.plan.facts.score).unsqueeze(-1),
                advantage_estimator=self.algorithm.advantage_estimator,
                step_wise=False,
            )
        )
        self.diagnostics.timings["load_worker_batch"] = max(output.load_seconds for output in loaded)

    async def forward(self, *, step_wall: StepWallTime | None) -> None:
        await self.context.wait(asyncio.gather(*self.policy.async_run_ray_method("pass_through", "barrier_all")))
        with Timer("fwd_logprobs_values_reward", self.diagnostics.timings):
            if self.colocate_all:
                await asyncio.to_thread(self.policy.backload_to_gpu, backload_optimizer=False, backload_model=True)
            self._pending = self.policy.async_run_ray_method("pass_through", "forward_loaded", self.plan.batch_id)
            await self.context.wait(asyncio.gather(*self._pending))
            self._pending = []
            if self.colocate_all:
                await asyncio.to_thread(self.policy.offload_to_cpu, offload_optimizer=False, offload_model=True)
            else:
                await self.context.wait(
                    asyncio.gather(*self.policy.async_run_ray_method("pass_through", "empty_cache"))
                )

    async def finalize(self, *, step_wall: StepWallTime | None) -> None:
        if step_wall is not None:
            step_wall.start("advantages")
        with Timer("prepare_worker_training_input", self.diagnostics.timings):
            prepared = await self.context.wait(
                asyncio.gather(*self.policy.async_run_ray_method("pass_through", "prepare_loaded", self.plan.batch_id))
            )
        self.diagnostics.metrics.update(prepared[0].metadata["batch_metrics"])

    def summary(self) -> BatchSummary:
        facts = self.plan.facts
        return BatchSummary(
            ConsumedWork(len(self.plan.uids), int(facts.response_len.sum()), int(facts.loss_tokens.sum())),
            len(self.metadata.groups),
            consumed_stop_metrics(facts.stop_reasons, len(self.plan.uids)),
            self.diagnostics.callback_rewards,
            self.plan.uids,
            self.plan.rollout_staleness,
            tuple(facts.response_len.tolist()),
            tuple(self.plan.global_step - group.policy_step for group in self.metadata.groups),
        )

    async def train(self, *, step_wall: StepWallTime | None) -> dict:
        if step_wall is not None:
            step_wall.start("policy_training")
        with (
            Timer("train_critic_and_policy", self.diagnostics.timings),
            critical_phase("train_step", self.plan.global_step),
        ):
            return await self.optimize(
                PolicyUpdate(
                    lambda: self.policy.async_run_ray_method("pass_through", "train_loaded", self.plan.batch_id), None
                ),
                self.diagnostics,
            )

    async def release(self, error: BaseException | None = None) -> None:
        try:
            async with asyncio.timeout(30):
                if error is not None:
                    for ref in self._pending:
                        ray.cancel(ref, force=False)
                if self._pending:
                    await asyncio.gather(*self._pending, return_exceptions=error is not None)
                if self.plan is not None:
                    await asyncio.gather(
                        *self.policy.async_run_ray_method("pass_through", "unload_batch", self.plan.batch_id)
                    )
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Loaded-batch cleanup failed: {cleanup_error}")
        finally:
            self._pending = []
            self._references = []
            self.plan = None
            self.diagnostics = None
