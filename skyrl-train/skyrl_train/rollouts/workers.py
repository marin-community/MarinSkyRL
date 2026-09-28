"""Rollout workers: processes that each host one trajectory runner, and the pool the trainer uses as its runner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import ray
from loguru import logger
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import PreTrainedTokenizerBase

from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.types import TrainingPhase, TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.utils.fd_monitor import start_fd_monitor
from skyrl_train.worker_setup import configure_worker_process

# Each worker imports its runner stack and loads its tokenizer from a shared filesystem; spacing the starts keeps
# those page-ins from overlapping each other and the engines' weight loads.
WORKER_START_INTERVAL_SECONDS = 2.0


class RolloutWorkers(Protocol):
    """The rollout processes a coordinator hands tasks to."""

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        """Generate and write one group, returning its response token count."""
        ...


@dataclass(frozen=True)
class WorkerShard:
    """A worker's position in its pool, for runners that divide per-process limits among the workers."""

    index: int
    count: int


class RunnerSpec(Protocol):
    """Picklable inputs that build a trajectory runner inside a rollout worker process."""

    config: DictConfig

    def build(self, tokenizer: PreTrainedTokenizerBase, shard: WorkerShard) -> TrajectoryRunner: ...


@dataclass(frozen=True)
class RolloutWorkerResources:
    """Size of a rollout worker pool, and how long a worker may go without completing a request."""

    num_workers: int
    cpus_per_worker: int
    executor_threads: int
    progress_timeout_seconds: float

    @classmethod
    def from_config(cls, config: DictConfig) -> RolloutWorkerResources:
        workers = config.trajectory_runner.rollout_workers
        return cls(
            num_workers=int(workers.num_workers),
            cpus_per_worker=int(workers.cpus_per_worker),
            executor_threads=int(workers.executor_threads),
            progress_timeout_seconds=float(workers.progress_timeout_seconds),
        )


class RolloutWorkerStalledError(TimeoutError):
    """A rollout worker completed no request before its progress deadline."""


@ray.remote
class RolloutWorker:
    """One rollout process; its event loop runs many requests concurrently on one trajectory runner."""

    def __init__(self, spec: RunnerSpec, shard: WorkerShard, sink: RetentionSink | None, executor_threads: int):
        configure_worker_process()
        start_fd_monitor()
        self._executor_threads = executor_threads
        self._runner = spec.build(tokenizer_from_config(spec.config), shard)
        if sink is not None:
            self._runner.set_trajectory_sink(sink)

    async def startup(self) -> None:
        # Harbor's litellm client runs each request's synchronous preamble on the loop's default executor, whose
        # default width would serialize the worker's concurrent requests.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=self._executor_threads, thread_name_prefix="rollout-worker")
        )
        await self._runner.startup()

    async def shutdown(self) -> None:
        await self._runner.shutdown()

    async def run(self, input_batch: TrajectoryRequestBatch) -> TrajectoryBatch:
        return await self._runner.run(input_batch, disable_tqdm=True)

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        return await self._runner.run_task(task, writer)

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None,
    ) -> None:
        await self._runner.start_eval_session(run_name=run_name, eval_step=eval_step, val_set_name=val_set_name)

    async def stop_eval_session(self) -> None:
        await self._runner.stop_eval_session()


class RolloutWorkerPool:
    """Rollout worker actors that the trainer uses as its trajectory runner.

    Each training task or ``run`` request goes whole to the least-loaded worker; training tasks write their groups
    straight to the rollout buffer. An evaluation session reserves worker 0, because a Harbor runner in an
    evaluation session sends every request to its evaluation orchestrator; evaluation requests run there, and
    training continues on the other workers, or waits when there is only one. A request fails with
    ``RolloutWorkerStalledError`` when its worker completes nothing for the progress timeout.

    Workers run on the driver's node, beside the rollout buffer actor they commit to and the Harbor proxy whose
    node-local log they read. They start one at a time.
    """

    def __init__(self, spec: RunnerSpec, resources: RolloutWorkerResources):
        self._spec = spec
        self._resources = resources
        self._sink: RetentionSink | None = None
        self._actors: list = []
        # Ray async actors accept every request concurrently, and a runner may queue them behind its own limits, so
        # a worker is stalled only when none of its requests completes, not when one request is slow.
        self._pending = [0] * resources.num_workers
        self._last_progress: list[float | None] = [None] * resources.num_workers
        self._routing = asyncio.Condition()
        self._eval_session_active = False

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        """Retain trajectories inside each worker, where its runner produces them.

        Workers receive the sink when they start, so a different sink cannot be attached afterwards.
        """
        if self._actors and sink is not self._sink:
            raise RuntimeError("attach the trajectory sink before the rollout workers start")
        self._sink = sink

    async def startup(self) -> None:
        node = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
        worker = RolloutWorker.options(num_cpus=self._resources.cpus_per_worker, scheduling_strategy=node)
        count = self._resources.num_workers
        for index in range(count):
            if index:
                await asyncio.sleep(WORKER_START_INTERVAL_SECONDS)
            actor = worker.remote(self._spec, WorkerShard(index, count), self._sink, self._resources.executor_threads)
            await actor.startup.remote()
            self._actors.append(actor)
            logger.info("Rollout worker {}/{} started", index + 1, count)

    async def shutdown(self) -> None:
        actors, self._actors = self._actors, []
        # Kill every worker before reporting any that failed to shut down cleanly.
        results = await asyncio.gather(*(actor.shutdown.remote() for actor in actors), return_exceptions=True)
        for actor in actors:
            ray.kill(actor)
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("rollout worker shutdown failed", errors)

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        del disable_tqdm
        return await self._dispatch(_training_phase(input_batch), lambda actor: actor.run.remote(input_batch))

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        return await self._dispatch(_training_phase(task.request), lambda actor: actor.run_task.remote(task, writer))

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None = None,
    ) -> None:
        """Reserve worker 0 once its training requests finish, and start the evaluation session on it."""
        async with self._routing:
            if self._eval_session_active:
                raise RuntimeError("an eval session is already active")
            self._eval_session_active = True
            await self._routing.wait_for(lambda: self._pending[0] == 0)
        try:
            await self._actors[0].start_eval_session.remote(
                run_name=run_name, eval_step=eval_step, val_set_name=val_set_name
            )
        except BaseException:
            await self._release_eval_worker()
            raise

    async def stop_eval_session(self) -> None:
        try:
            await self._actors[0].stop_eval_session.remote()
        finally:
            await self._release_eval_worker()

    async def _release_eval_worker(self) -> None:
        async with self._routing:
            self._eval_session_active = False
            self._routing.notify_all()

    def _training_worker(self) -> int | None:
        """The least-loaded worker that may take training work now, if any."""
        reserved = {0} if self._eval_session_active else set()
        eligible = [index for index in range(len(self._actors)) if index not in reserved]
        return min(eligible, key=self._pending.__getitem__, default=None)

    async def _dispatch(self, phase: TrainingPhase, submit: Callable[[Any], Awaitable[Any]]) -> Any:
        async with self._routing:
            if phase == "eval":
                if not self._eval_session_active:
                    raise RuntimeError("evaluation request received without an active eval session")
                index = 0
            else:
                await self._routing.wait_for(lambda: self._training_worker() is not None)
                index = self._training_worker()
            loop = asyncio.get_running_loop()
            if self._pending[index] == 0:
                self._last_progress[index] = loop.time()
            self._pending[index] += 1
        try:
            return await self._await_progress(index, submit(self._actors[index]))
        finally:
            async with self._routing:
                self._pending[index] -= 1
                if self._pending[index] == 0:
                    self._last_progress[index] = None
                self._routing.notify_all()

    async def _await_progress(self, index: int, request: Any) -> Any:
        """Wait for one request, failing it when its worker completes nothing for the progress timeout."""
        loop = asyncio.get_running_loop()
        result = asyncio.ensure_future(request)
        try:
            while True:
                observed = self._last_progress[index]
                assert observed is not None
                deadline = asyncio.timeout_at(observed + self._resources.progress_timeout_seconds)
                try:
                    async with deadline:
                        output = await asyncio.shield(result)
                except TimeoutError as error:
                    if not deadline.expired():
                        raise
                    if self._last_progress[index] != observed:
                        # Another request on this worker completed while this one waited, so the worker is live.
                        continue
                    # Cancel the remote request too, so the runner unwinds its work instead of leaving it detached.
                    ray.cancel(request, force=False, recursive=True)
                    raise RolloutWorkerStalledError(
                        f"rollout worker {index} completed no request for {self._resources.progress_timeout_seconds:g}s"
                    ) from error
                self._last_progress[index] = loop.time()
                return output
        finally:
            # An abandoned request's result would otherwise surface later as an unretrieved error.
            result.cancel()


def _training_phase(input_batch: TrajectoryRequestBatch) -> TrainingPhase:
    metadata = input_batch.get("batch_metadata")
    return metadata.training_phase if metadata is not None else "train"
