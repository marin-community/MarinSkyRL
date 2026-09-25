"""Independent synchronous rollout production and batch delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.rollout_buffer import Rollout, RolloutBuffer, RolloutDataLoader, RolloutRequest
from skyrl_train.rollout_worker import RolloutWorker
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request


@dataclass(frozen=True)
class SynchronousRolloutBatch:
    request: RolloutRequest
    rollouts: list[Rollout]


class SynchronousCurriculum:
    """Own the dataset cursor and form one assignment after buffer capacity opens."""

    def __init__(
        self,
        dataloader: Iterable[list[dict]],
        *,
        epochs: int,
        start_epoch: int,
        select_prompts: Callable[[list[dict]], list[dict]],
        samples_per_prompt: int,
        backend: str,
        sampling_params: object,
        environment_class: str,
        current_step: Callable[[], int],
    ):
        self.dataloader = dataloader
        self.epochs = epochs
        self.epoch = start_epoch
        self.iterator = iter(dataloader)
        self.select_prompts = select_prompts
        self.samples_per_prompt = samples_per_prompt
        self.backend = backend
        self.sampling_params = sampling_params
        self.environment_class = environment_class
        self.current_step = current_step

    async def next_assignment(self) -> RolloutRequest | None:
        try:
            entries = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if self.epoch < self.epochs:
                self.iterator = iter(self.dataloader)
            return None
        prompts = self.select_prompts(entries)
        step = self.current_step()
        trajectory_request, uids = prepare_trajectory_request(
            prompts,
            self.samples_per_prompt,
            get_sampling_params_for_backend(self.backend, self.sampling_params),
            self.environment_class,
            "train",
            step,
        )
        return RolloutRequest(trajectory_request, prompts, uids, step, "batch")


class SynchronousRolloutBuffer:
    """Release one curriculum assignment per completed and consumed training batch."""

    def __init__(self, storage: RolloutBuffer, retain: Callable[[Rollout], Awaitable[None]]):
        self.storage = storage
        self.retain = retain
        self._capacity = asyncio.Event()
        self._ready = asyncio.Event()
        self._request: RolloutRequest | None = None
        self._count = 0
        self._error: BaseException | None = None
        self._epoch_end = False

    async def wait_for_slot(self) -> None:
        await self._capacity.wait()
        self._capacity.clear()

    def open_slot(self) -> None:
        """Let the producer start after the learner has prepared this step's inference state."""
        self._capacity.set()

    def begin(self, request: RolloutRequest) -> None:
        self._request = request
        self._epoch_end = False
        self._ready.clear()

    def complete(self, count: int) -> None:
        self._count = count
        self._ready.set()

    def fail(self, error: BaseException) -> None:
        self._error = error
        self._ready.set()

    def finish_epoch(self) -> None:
        self._epoch_end = True
        self._ready.set()

    async def next_batch(self) -> SynchronousRolloutBatch | None:
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        if self._epoch_end:
            return None
        if self._request is None:
            raise RuntimeError("synchronous rollout buffer has no active assignment")
        rollouts = await self.storage.next_batch(self._count)
        if len(rollouts) != self._count:
            raise RuntimeError(f"buffer returned {len(rollouts)} of {self._count} completed rollouts")
        for rollout in rollouts:
            await self.retain(rollout)
        return SynchronousRolloutBatch(self._request, rollouts)

    def acknowledge(self) -> None:
        self._request = None
        self._count = 0
        self._ready.clear()

    def acknowledge_epoch(self) -> None:
        self._epoch_end = False
        self._ready.clear()


class SynchronousRolloutPipeline:
    """Run the curriculum and worker independently of the learner loop."""

    def __init__(self, curriculum: RolloutDataLoader, buffer: SynchronousRolloutBuffer, worker: RolloutWorker):
        self.curriculum = curriculum
        self.buffer = buffer
        self.worker = worker
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                await self.buffer.wait_for_slot()
                request = await self.curriculum.next_assignment()
                if request is None:
                    self.buffer.finish_epoch()
                    continue
                self.buffer.begin(request)
                count = await self.worker.produce(request)
                self.buffer.complete(count)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.buffer.fail(error)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
