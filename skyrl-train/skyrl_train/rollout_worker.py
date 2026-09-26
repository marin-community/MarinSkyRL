"""Synchronous-trainer workers that commit and publish completed rollout batches."""

from __future__ import annotations

from typing import Protocol

from skyrl_train.rollout_buffer import (
    RolloutBuffer,
    RolloutReceipt,
    RolloutRequest,
    RolloutWriter,
    SynchronousRollout,
)
from skyrl_train.trajectory_runners.base import TrajectoryRunner


class RolloutExecutor(Protocol):
    async def produce(self, request: RolloutRequest, disable_tqdm: bool = False) -> list[RolloutReceipt]: ...


class RolloutWorker:
    """Reserve buffer capacity and publish committed rollout records."""

    def __init__(self, executor: RolloutExecutor, buffer: RolloutBuffer):
        self.executor = executor
        self.buffer = buffer

    async def produce(self, request: RolloutRequest, disable_tqdm: bool = False) -> int:
        async with self.buffer.request_slot() as slot:
            receipts = await self.executor.produce(request, disable_tqdm=disable_tqdm)
            for receipt in receipts:
                await slot.publish(receipt)
            return len(receipts)


class LocalRolloutWorker:
    """Run a harness in this process and write its completed training record."""

    def __init__(self, runner: TrajectoryRunner, writer: RolloutWriter):
        self.runner = runner
        self.writer = writer

    async def produce(self, request: RolloutRequest, disable_tqdm: bool = False) -> list[RolloutReceipt]:
        output = await self.runner.run(request.trajectory_request, disable_tqdm=disable_tqdm)
        rollout = SynchronousRollout(
            output, request.uids, request.source_prompts, request.model_step, request_batch=request.trajectory_request
        )
        return [await self.writer.write_rollout(rollout)]


def bind_rollout_worker(runner: TrajectoryRunner, buffer: RolloutBuffer) -> RolloutWorker:
    """Bind a harness to the synchronous training buffer."""
    return RolloutWorker(LocalRolloutWorker(runner, buffer.writer()), buffer)
