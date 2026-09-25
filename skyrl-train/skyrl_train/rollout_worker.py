"""Training workers that commit and publish completed trajectories."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Protocol

from loguru import logger

from skyrl_train.async_rollout_state import GeneratedOutputGroup
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollout_buffer import (
    Rollout,
    RolloutBuffer,
    RolloutDataLoader,
    RolloutReceipt,
    RolloutRequest,
    RolloutSlot,
    RolloutWriter,
    SynchronousRollout,
)
from skyrl_train.trajectory_runners.base import TrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.utils.logging_utils import log_exception_as_text
from skyrl_train.telemetry import record_rollout_buffer


class GenerationStalledError(RuntimeError):
    """The task source has no more work and no retry arrived before its deadline."""


class RolloutTaskDataLoader(Protocol):
    async def get_next_non_consumed_data(self) -> list[dict] | None: ...


class AsyncRolloutTaskSource:
    """Give continuous workers one prompt group at a time, including retries."""

    def __init__(
        self,
        dataloader: RolloutTaskDataLoader,
        retries: asyncio.Queue[list[dict]],
        *,
        samples_per_prompt: int,
        backend: str,
        sampling_params: object,
        environment_class: str,
        current_step: Callable[[], int],
        stall_timeout: Callable[[], float],
    ):
        self.dataloader = dataloader
        self.retries = retries
        self.samples_per_prompt = samples_per_prompt
        self.backend = backend
        self.sampling_params = sampling_params
        self.environment_class = environment_class
        self.current_step = current_step
        self.stall_timeout = stall_timeout

    async def next_prompts(self) -> list[dict]:
        try:
            return self.retries.get_nowait()
        except asyncio.QueueEmpty:
            prompts = await self.dataloader.get_next_non_consumed_data()
            if prompts is not None:
                return prompts
        try:
            return await asyncio.wait_for(self.retries.get(), timeout=self.stall_timeout())
        except asyncio.TimeoutError as error:
            raise GenerationStalledError(
                "Dataset exhausted and no retries arrived within the stall deadline"
            ) from error

    async def next_assignment(self) -> RolloutRequest:
        prompts = await self.next_prompts()
        step = self.current_step()
        trajectory_request, uids = prepare_trajectory_request(
            prompts,
            self.samples_per_prompt,
            get_sampling_params_for_backend(self.backend, self.sampling_params),
            self.environment_class,
            "train",
            step,
        )
        if len(set(uids)) != 1:
            raise ValueError("one grouped rollout must contain exactly one UID")
        return RolloutRequest(trajectory_request, prompts, uids, step, "group")


class RolloutExecutor(Protocol):
    async def produce(self, request: RolloutRequest, disable_tqdm: bool = False) -> list[RolloutReceipt]: ...

    async def retain(self, rollout: Rollout) -> None: ...


class RolloutWorker:
    """Reserve buffer capacity and publish committed rollout records."""

    def __init__(
        self,
        executor: RolloutExecutor,
        buffer: RolloutBuffer,
        ready_condition: asyncio.Condition | None = None,
    ):
        self.executor = executor
        self.buffer = buffer
        self.ready_condition = ready_condition

    async def produce(self, request: RolloutRequest, disable_tqdm: bool = False) -> int:
        async with self.buffer.request_slot() as slot:
            return await self._produce(slot, request, disable_tqdm)

    async def produce_next(self, source: RolloutDataLoader) -> int:
        async with self.buffer.request_slot() as slot:
            request = await source.next_assignment()
            if request is None:
                raise GenerationStalledError("rollout source is exhausted")
            return await self._produce(slot, request, disable_tqdm=True)

    async def _produce(self, slot: RolloutSlot, request: RolloutRequest, disable_tqdm: bool) -> int:
        receipts = await self.executor.produce(request, disable_tqdm=disable_tqdm)
        if request.kind == "group" and len(receipts) != 1:
            raise ValueError("one async prompt must produce one buffered reward group")
        if self.ready_condition is None:
            for receipt in receipts:
                await slot.publish(receipt)
            return len(receipts)
        async with self.ready_condition:
            for receipt in receipts:
                while self.buffer.full():
                    await self.ready_condition.wait()
                await slot.publish(receipt)
            self.ready_condition.notify_all()
        return len(receipts)

    async def retain(self, rollout: Rollout) -> None:
        await self.executor.retain(rollout)


class AsyncRolloutWorkerPool:
    """Keep N independent prompt groups in generation while the trainer reads the buffer."""

    def __init__(
        self,
        worker: RolloutWorker,
        source: RolloutDataLoader,
        count: int,
        on_worker_finished: Callable[[], Awaitable[None]],
    ):
        self.worker = worker
        self.source = source
        self.count = count
        self.on_worker_finished = on_worker_finished

    def start(self) -> list[asyncio.Task]:
        return [asyncio.create_task(self._serve()) for _ in range(self.count)]

    async def _serve(self) -> None:
        try:
            while True:
                await self.worker.produce_next(self.source)
                record_rollout_buffer(self.worker.buffer.pending_count(), self.worker.buffer.capacity())
        except asyncio.CancelledError:
            return
        except GenerationStalledError:
            logger.info("Trajectory worker exiting: collection stalled (dataset exhausted, no retries)")
            return
        except Exception as error:
            log_exception_as_text("Trajectory worker failed", error)
            sys.exit(1)
        finally:
            await self.on_worker_finished()


class LocalRolloutWorker:
    """Run a harness in this process and write its completed training record."""

    def __init__(self, runner: TrajectoryRunner, writer: RolloutWriter):
        self.runner = runner
        self.writer = writer

    async def produce(self, request: RolloutRequest, disable_tqdm: bool = False) -> list[RolloutReceipt]:
        output = await self.runner.run(request.trajectory_request, disable_tqdm=disable_tqdm)
        actual_step = output.get("actual_global_step")
        sampled_step = actual_step if actual_step is not None else request.model_step
        if request.kind == "group":
            if len(set(request.uids)) != 1:
                raise ValueError("one grouped rollout must contain exactly one UID")
            rollout = GeneratedOutputGroup(
                output, request.uids[0], sampled_step, request.source_prompts, request_batch=request.trajectory_request
            )
        else:
            rollout = SynchronousRollout(
                output, request.uids, request.source_prompts, sampled_step, request_batch=request.trajectory_request
            )
        return [await self.writer.write_rollout(rollout)]

    async def retain(self, rollout: Rollout) -> None:
        # Local runners retain during output finalization, before committing.
        pass


def bind_rollout_worker(
    runner: TrajectoryRunner,
    buffer: RolloutBuffer,
    ready_condition: asyncio.Condition | None = None,
) -> RolloutWorker:
    """Bind a harness to the training buffer without changing its public API."""
    return RolloutWorker(bind_rollout_executor(runner, buffer), buffer, ready_condition)


def bind_rollout_executor(runner: TrajectoryRunner, buffer: RolloutBuffer) -> RolloutExecutor:
    """Choose execution placement for one producer."""
    from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import HarborRolloutWorker, RolloutDispatcher
    from skyrl_train.trajectory_runners.nemotron_ultra import NemotronUltraRolloutWorker, NemotronUltraTrajectoryRouter

    if isinstance(runner, RolloutDispatcher):
        return HarborRolloutWorker(runner, buffer.remote_writer())
    if isinstance(runner, NemotronUltraTrajectoryRouter):
        return NemotronUltraRolloutWorker(runner, buffer)
    return LocalRolloutWorker(runner, buffer.writer())
