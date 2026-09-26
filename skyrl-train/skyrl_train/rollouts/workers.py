"""Rollout workers: processes that generate one prompt group per task and write it to the rollout buffer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import ray
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from skyrl_train.rollouts.buffer import RolloutTask, RolloutWriter
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.trajectory_runners.types import TrajectoryBatch, TrajectoryRequestBatch
from skyrl_train.worker_setup import configure_worker_process


class RolloutWorkers(Protocol):
    """The rollout processes a coordinator hands tasks to."""

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        """Generate and write one group, returning its response token count."""
        ...


class RunnerSpec(Protocol):
    """Picklable inputs that build a trajectory runner inside a rollout worker process."""

    config: DictConfig

    def build(self, tokenizer: PreTrainedTokenizerBase) -> TrajectoryRunner: ...


@ray.remote
class RolloutWorker:
    """One rollout process; its event loop runs many tasks concurrently."""

    def __init__(self, spec: RunnerSpec, sink: RetentionSink | None):
        configure_worker_process()
        self._runner = spec.build(tokenizer_from_config(spec.config))
        if sink is not None:
            self._runner.set_trajectory_sink(sink)

    async def startup(self) -> None:
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
        n_concurrent_trials: int | None,
    ) -> None:
        await self._runner.start_eval_session(
            run_name=run_name,
            eval_step=eval_step,
            val_set_name=val_set_name,
            n_concurrent_trials=n_concurrent_trials,
        )

    async def stop_eval_session(self) -> None:
        await self._runner.stop_eval_session()


class RolloutWorkerPool:
    """Rollout worker actors behind least-loaded dispatch.

    The pool is also the trainer's trajectory runner: evaluation requests run on a worker and return to the
    caller, while training tasks write straight to the buffer.
    """

    def __init__(self, spec: RunnerSpec, *, num_workers: int, cpus_per_worker: int):
        if num_workers < 1 or cpus_per_worker < 1:
            raise ValueError("a rollout worker pool needs at least one worker and one CPU per worker")
        self._spec = spec
        self._num_workers = num_workers
        self._cpus_per_worker = cpus_per_worker
        self._sink: RetentionSink | None = None
        self._actors: list = []
        self._load: list[int] = []

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        """Retain trajectories inside each worker, where its runner produces them."""
        self._sink = sink

    async def startup(self) -> None:
        worker = RolloutWorker.options(num_cpus=self._cpus_per_worker)
        self._actors = [worker.remote(self._spec, self._sink) for _ in range(self._num_workers)]
        self._load = [0] * self._num_workers
        await asyncio.gather(*(actor.startup.remote() for actor in self._actors))

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
        return await self._submit(lambda actor: actor.run.remote(input_batch))

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        return await self._submit(lambda actor: actor.run_task.remote(task, writer))

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None = None,
        n_concurrent_trials: int | None = None,
    ) -> None:
        await asyncio.gather(
            *(
                actor.start_eval_session.remote(
                    run_name=run_name,
                    eval_step=eval_step,
                    val_set_name=val_set_name,
                    n_concurrent_trials=n_concurrent_trials,
                )
                for actor in self._actors
            )
        )

    async def stop_eval_session(self) -> None:
        await asyncio.gather(*(actor.stop_eval_session.remote() for actor in self._actors))

    async def _submit(self, call: Callable[[Any], Awaitable[Any]]) -> Any:
        index = min(range(len(self._actors)), key=self._load.__getitem__)
        self._load[index] += 1
        try:
            return await call(self._actors[index])
        finally:
            self._load[index] -= 1
