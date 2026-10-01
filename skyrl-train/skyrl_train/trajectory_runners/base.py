from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
from types import MappingProxyType
from skyrl_train.trajectory_runners.types import (
    BatchMetadata as BatchMetadata,
    ConversationType as ConversationType,
    TrajectoryBatch as TrajectoryBatch,
    TrajectoryID as TrajectoryID,
    TrajectoryRequestBatch as TrajectoryRequestBatch,
    TrainingPhase as TrainingPhase,
)
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutTask, RolloutWriter
from skyrl_train.trajectory_runners.trajectory_retention import RetentionSink
from skyrl_train.rollout_observability import rollout_wait
from skyrl_train.rollouts.finalization import finalize_trajectory_batch


class BatchRunner(Protocol):
    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch: ...


async def run_rollout_task(runner: BatchRunner, task: RolloutTask, writer: RolloutWriter) -> int:
    """Generate one leased prompt group, write it to the rollout buffer, and return its response token count."""
    output = await runner.run(task.request, disable_tqdm=True)
    group = RolloutGroup(output, task.prompt["uid"], task.lease.policy_step, task.prompt)
    with rollout_wait("enqueue"):
        await writer.write_rollout(task.lease, group)
    return sum(len(response) for response in output["response_ids"])


class TrajectoryRunner(ABC):
    """Abstract base class for acquiring trainer-ready trajectories.

    Lifecycle:
        1. __init__() - Synchronous initialization (no async resources)
        2. startup() - Async initialization of resources (e.g., orchestrators, connections)
        3. run() - Called repeatedly during training
        4. shutdown() - Async cleanup of resources

    Implementations should handle errors gracefully in run() to avoid killing the
    training job. Use restart logic for recoverable failures.
    """

    trajectory_runner_cfg = MappingProxyType({})
    trajectory_sink: RetentionSink | None = None

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Acquire trajectories and apply runner-independent output finalization.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (TrajectoryRequestBatch): Input batch
        Returns:
            TrajectoryBatch: Generated trajectories
        """
        output = await self._run(input_batch, disable_tqdm=disable_tqdm)
        return await finalize_trajectory_batch(input_batch, output, self.trajectory_runner_cfg, self.trajectory_sink)

    async def run_task(self, task: RolloutTask, writer: RolloutWriter) -> int:
        return await run_rollout_task(self, task, writer)

    def set_trajectory_sink(self, sink: RetentionSink) -> None:
        """Attach the trainer-owned sink used by shared output finalization."""
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def start_eval_session(
        self,
        *,
        run_name: str,
        eval_step: int,
        val_set_name: str | None = None,
    ) -> None:
        """Start an evaluation-scoped resource session when a runner needs one."""

    async def stop_eval_session(self) -> None:
        """Stop resources created for the current evaluation session."""

    @abstractmethod
    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Produce trajectories before shared output finalization."""
        raise NotImplementedError()

    async def startup(self) -> None:
        """Initialize runner resources before the first call to :meth:`run`."""
        pass

    async def shutdown(self) -> None:
        """Release runner resources after use; repeated calls must be safe."""
        pass
