"""Rollout state of one training run and the coordinator loop that feeds its buffer."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, TypeVar

import ray
from loguru import logger
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from marinskyrl.environment_contract import TrainingType
from skyrl_train.curriculum import CurriculumConfig, CurriculumOrder, SamplingKind
from skyrl_train.dataset import PromptDataset
from skyrl_train.domain_sampling import DomainWeightedOrder
from skyrl_train.dynamic_sampling import (
    DynamicSamplingType,
    GroupSelectionPolicy,
    resolve_dynamic_sampling_criteria,
)
from skyrl_train.group_admission import GroupAdmissionPolicy, GroupAdmissionStalledError, GroupAdvantageInvariant
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollouts.buffer import (
    BufferSnapshot,
    MemoryRolloutWriter,
    ReadyRollout,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutGroup,
    RolloutTask,
)
from skyrl_train.rollouts.loader import GroupLoader, GroupLoaderState, PromptGroupDataset, PromptOrder, SeededPasses
from skyrl_train.rollout_observability import dispatch_wait, observe_rollout_call, record_group_disposition
from skyrl_train.rollouts.workers import RolloutWorkers
from skyrl_train.telemetry import record_generated_work, record_rollout_buffer
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.trajectory_runners.types import TrajectoryRequestBatch
from skyrl_train.utils.algorithm_registry import policy_loss_requires_rollout_logprobs

_T = TypeVar("_T")


@dataclass(frozen=True)
class RolloutRequestSpec:
    """How the coordinator turns one prompt group into a trajectory request."""

    samples_per_prompt: int
    sampling_params: dict
    environment_class: str

    @classmethod
    def from_config(cls, config: DictConfig) -> RolloutRequestSpec:
        return cls(
            samples_per_prompt=config.generator.n_samples_per_prompt,
            sampling_params=get_sampling_params_for_backend(config.generator.backend, config.generator.sampling_params),
            environment_class=config.environment.env_class,
        )

    def request(self, prompt: dict, policy_step: int) -> TrajectoryRequestBatch:
        request, _ = prepare_trajectory_request(
            [prompt], self.samples_per_prompt, self.sampling_params, self.environment_class, "train", policy_step
        )
        return request


@dataclass(frozen=True)
class TrainingContextState:
    """Checkpointed rollout state: the loader, including prompts to regenerate, and committed groups.

    Each ready rollout's ``payload`` holds its ``RolloutGroup`` rather than an object reference.
    """

    loader: GroupLoaderState
    ready: list[ReadyRollout]


def prompt_order_from_config(config: DictConfig, dataset: PromptGroupDataset) -> PromptOrder:
    """Build the loader's prompt order: seeded passes by default, or the ``data.sampling`` order."""
    sampling = config.data.sampling
    if sampling.kind is None:
        return SeededPasses(len(dataset), seed=config.trainer.seed, shuffle=config.data.shuffle)
    if config.trainer.step_wise_training:
        raise ValueError("data.sampling.kind requires one group per prompt; step_wise_training is not supported")
    if not isinstance(dataset, PromptDataset):
        raise ValueError(f"data.sampling.kind requires a prompt dataset, got {type(dataset).__name__}")
    seed = sampling.seed if sampling.seed is not None else config.trainer.seed
    batch_size = config.trainer.train_batch_size
    if SamplingKind(sampling.kind) is SamplingKind.DOMAIN_WEIGHTED:
        return DomainWeightedOrder(dataset.dataframe, sampling.domain_weights, seed=seed, window_size=batch_size)
    curriculum = CurriculumConfig.from_dict_config(sampling, group_size=config.generator.n_samples_per_prompt)
    return CurriculumOrder(dataset, curriculum, seed=seed, window_size=batch_size)


class TrainingContext:
    """The group loader, the rollout buffer, and the rollout tasks in flight between them.

    ``start`` runs the coordinator loop: for every lease the buffer grants, it takes the next prompt group
    from the loader and hands one task to a rollout worker, which writes the result to the buffer. A failed
    task fails training: the next ``next_batch`` or ``publish`` raises its error. With ``rollout_spans`` it
    records each rollout call, the loop's waits, and every group's disposition.
    """

    def __init__(
        self,
        loader: GroupLoader,
        config: RolloutBufferConfig,
        content_policy: RolloutContentPolicy,
        request_spec: RolloutRequestSpec,
        workers: RolloutWorkers,
        *,
        rollout_spans: bool,
    ):
        self.loader = loader
        self.config = config
        self._request_spec = request_spec
        self._workers = workers
        self._rollout_spans = rollout_spans
        self.mode = TrainingType.SYNC if config.max_staleness_steps == 0 else TrainingType.ASYNC
        self._policy_step = 0
        # The trainer reads every payload through this actor; keep it beside the trainer.
        node = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
        self._buffer = ray.remote(RolloutBuffer).options(num_cpus=0, scheduling_strategy=node).remote(config)
        self._writer = MemoryRolloutWriter(self._buffer, content_policy)
        self._in_flight: dict[str, RolloutTask] = {}
        self._running: set[asyncio.Task] = set()
        self._dispatcher: asyncio.Task | None = None
        self._failure: asyncio.Future | None = None

    @classmethod
    def from_config(cls, config: DictConfig, dataset: PromptGroupDataset, workers: RolloutWorkers) -> TrainingContext:
        algorithm = config.trainer.algorithm
        dynamic_sampling = algorithm.dynamic_sampling
        selection = GroupSelectionPolicy(
            DynamicSamplingType(dynamic_sampling.type) if dynamic_sampling.type is not None else None,
            criteria=resolve_dynamic_sampling_criteria(
                dynamic_sampling.informative_on, float(dynamic_sampling.min_reward_std)
            ),
        )
        batch_size = config.trainer.train_batch_size
        max_sample_batches = int(dynamic_sampling.max_sample_batches)
        buffer_config = RolloutBufferConfig(
            batch_size=batch_size,
            max_in_flight=config.trainer.rollout_buffer.max_in_flight,
            max_staleness_steps=config.trainer.rollout_buffer.max_staleness_steps,
            dynamic_sampling=selection.sampling_type,
            max_candidate_groups=max_sample_batches * batch_size if max_sample_batches > 0 else None,
        )
        admission = GroupAdmissionPolicy(
            GroupAdvantageInvariant.from_config(algorithm.resolved_group_advantage),
            rollout_logprobs_required=policy_loss_requires_rollout_logprobs(algorithm.policy_loss_type),
        )
        return cls(
            GroupLoader(dataset, prompt_order_from_config(config, dataset)),
            buffer_config,
            RolloutContentPolicy(admission, selection),
            RolloutRequestSpec.from_config(config),
            workers,
            rollout_spans=config.trainer.rollout_spans,
        )

    def start(self) -> None:
        """Start dispatching; the buffer grants no lease before the first ``publish``."""
        self._failure = asyncio.get_running_loop().create_future()
        self._dispatcher = asyncio.create_task(self._dispatch())

    async def close(self) -> None:
        tasks = [task for task in (self._dispatcher, *self._running) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        ray.kill(self._buffer)

    async def publish(self, policy_step: int) -> None:
        """Acknowledge the previous batch and lease rollouts at ``policy_step``, whose weights are now live."""
        await self._until_failure(self._buffer.publish.remote(policy_step))
        self._policy_step = policy_step

    async def next_batch(
        self,
        *,
        stall_timeout: float,
        on_admitted: Callable[[list[RolloutGroup]], Awaitable[None]],
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Wait for the current step's batch and return its groups with selection and prompt-order metrics.

        Groups reach ``on_admitted`` as they are admitted, before the batch is complete. Every group judged for
        the batch, kept or discarded, then updates the loader's prompt order.

        Raises:
            GroupAdmissionStalledError: No group was admitted, or admitted payloads did not arrive, for
                ``stall_timeout`` seconds.
        """
        loop = asyncio.get_running_loop()
        groups: list[RolloutGroup] = []
        deadline = loop.time() + stall_timeout
        while True:
            timeout = max(deadline - loop.time(), 0.0)
            admission = await self._until_failure(self._buffer.admit.remote(timeout))
            for prompt in admission.retries:
                self.loader.retry(prompt)
            for policy_step, work in admission.generated:
                record_generated_work(work, policy_step)
            if self._rollout_spans:
                for outcome in admission.dispositions:
                    record_group_disposition(
                        disposition=outcome.disposition,
                        tokens=outcome.tokens,
                        step=self._policy_step,
                        dwell_seconds=outcome.dwell_seconds,
                    )
            record_rollout_buffer(admission.ready_count, self.config.max_untrained_groups)
            if admission.payloads:
                try:
                    async with asyncio.timeout(stall_timeout):
                        admitted = await self._until_failure(asyncio.gather(*admission.payloads))
                except TimeoutError as error:
                    raise GroupAdmissionStalledError(
                        f"{len(admission.payloads)} admitted rollout payloads did not arrive within "
                        f"{stall_timeout:.0f}s: policy_step={self._policy_step} admitted={len(groups)}"
                    ) from error
                groups.extend(admitted)
                await on_admitted(admitted)
                deadline = loop.time() + stall_timeout
            if admission.selection is not None:
                return groups, {**admission.selection.metrics, **self.loader.observe(admission.selection.judged)}

    async def state_dict(self) -> TrainingContextState:
        """Capture every dispatched group that no trained batch has consumed.

        The loader position and in-flight tasks are read together, before the buffer snapshot, so a task
        that commits meanwhile appears in the buffer and not also among the prompts to regenerate.
        """
        in_flight = dict(self._in_flight)
        loader = self.loader.state_dict()
        snapshot: BufferSnapshot = await self._buffer.snapshot.remote()
        uncommitted = [task.prompt for lease_id, task in in_flight.items() if lease_id in snapshot.leases]
        payloads = await asyncio.gather(*(asyncio.gather(*rollout.payload) for rollout in snapshot.ready))
        ready = [
            # The buffer's clock does not carry across processes.
            dataclasses.replace(rollout, payload=list(payload), committed_at=None)
            for rollout, payload in zip(snapshot.ready, payloads, strict=True)
        ]
        retries = [*loader.retries, *snapshot.retries, *uncommitted]
        return TrainingContextState(dataclasses.replace(loader, retries=retries), ready)

    async def load_state_dict(self, state: TrainingContextState) -> None:
        """Restore a checkpoint's rollout state before ``start``."""
        if self._dispatcher is not None:
            raise RuntimeError("rollout state must be restored before dispatching starts")
        self.loader.load_state_dict(state.loader)
        # Ray aborts the process when an object's owner is an actor that has not started yet.
        await self._buffer.__ray_ready__.remote()
        ready = [
            dataclasses.replace(rollout, payload=[ray.put(group, _owner=self._buffer) for group in rollout.payload])
            for rollout in state.ready
        ]
        await self._buffer.restore.remote(BufferSnapshot(ready=ready, retries=[], leases=frozenset()))
        logger.info(
            "Restored rollout state: {} committed groups and {} prompts to regenerate",
            len(ready),
            len(state.loader.retries),
        )

    async def _dispatch(self) -> None:
        try:
            while True:
                with self._wait("slot"):
                    lease = await self._buffer.acquire_lease.remote()
                with self._wait("prompt"):
                    prompt = self.loader.next_group()
                task = RolloutTask(lease, prompt, self._request_spec.request(prompt, lease.policy_step))
                self._in_flight[lease.lease_id] = task
                running = asyncio.create_task(self._run(task))
                self._running.add(running)
                running.add_done_callback(self._running.discard)
        except Exception as error:
            self._fail(error)

    def _wait(self, name: str) -> AbstractContextManager[None]:
        return dispatch_wait(name, step=self._policy_step, mode=self.mode, enabled=self._rollout_spans)

    async def _run(self, task: RolloutTask) -> None:
        try:
            with observe_rollout_call(
                step=task.lease.policy_step, mode=self.mode, enabled=self._rollout_spans
            ) as observation:
                response_tokens = await self._workers.run_task(task, self._writer)
                if observation is not None:
                    observation.response_tokens = response_tokens
        except Exception as error:
            self._fail(error)
        finally:
            del self._in_flight[task.lease.lease_id]

    def _fail(self, error: Exception) -> None:
        logger.opt(exception=error).error("Rollout dispatch failed; training will stop")
        assert self._failure is not None
        if not self._failure.done():
            self._failure.set_exception(error)

    async def _until_failure(self, awaitable: Awaitable[_T]) -> _T:
        """Await ``awaitable`` unless a rollout task fails first, then raise that failure."""
        assert self._failure is not None, "start the training context before reading from it"
        work: asyncio.Future[Any] = asyncio.ensure_future(awaitable)
        await asyncio.wait({work, self._failure}, return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            return work.result()
        work.cancel()
        return self._failure.result()
