from abc import ABC, abstractmethod
from types import MappingProxyType
from skyrl_train.trajectory_runners.types import (
    BatchMetadata as BatchMetadata,
    ConversationType as ConversationType,
    TrajectoryBatch as TrajectoryBatch,
    TrajectoryID as TrajectoryID,
    TrajectoryRequestBatch as TrajectoryRequestBatch,
    TrainingPhase as TrainingPhase,
    TIS_ALIGNED_TOKENS_METRIC,
    TIS_EXACT_MATCH_FRACTION_METRIC,
    TIS_LCS_FALLBACK_FRACTION_METRIC,
    TIS_UNALIGNED_FRACTION_METRIC,
    TIS_ALIGNMENT_FAIL_COUNT_METRIC,
    TIS_LCS_FALLBACK_MESSAGES_METRIC,
    TIS_LCS_FALLBACK_ALERT_METRIC,
)
from skyrl_train.trajectory_runners.trajectory_reward_shaping import shape_trajectory_rewards
from skyrl_train.trajectory_runners.trajectory_retention import TrajectorySink, retain_trajectories


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
    trajectory_sink: TrajectorySink | None = None

    async def run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Acquire trajectories and apply runner-independent output finalization.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (TrajectoryRequestBatch): Input batch
        Returns:
            TrajectoryBatch: Generated trajectories
        """
        output = await self._run(input_batch, disable_tqdm=disable_tqdm)
        return await self._finalize_output(input_batch, output)

    async def _finalize_output(self, input_batch: TrajectoryRequestBatch, output: TrajectoryBatch) -> TrajectoryBatch:
        """Apply runner-independent shaping, metrics, and retention."""
        shape_trajectory_rewards(output, self.trajectory_runner_cfg.get("trajectory_reward_shaping"))
        self._add_alignment_metrics(output)
        if self.trajectory_sink is not None:
            await retain_trajectories(self.trajectory_sink, input_batch, output)
        return output

    def set_trajectory_sink(self, sink: TrajectorySink) -> None:
        """Attach the trainer-owned sink used by shared output finalization."""
        sink.bind_runner(type(self).__name__)
        self.trajectory_sink = sink

    async def start_eval_session(self, *, run_name: str, eval_step: int, val_set_name: str | None = None) -> None:
        """Start an evaluation-scoped resource session when a runner needs one."""

    async def stop_eval_session(self) -> None:
        """Stop resources created for the current evaluation session."""

    @abstractmethod
    async def _run(self, input_batch: TrajectoryRequestBatch, disable_tqdm: bool = False) -> TrajectoryBatch:
        """Produce trajectories before shared output finalization."""
        raise NotImplementedError()

    @staticmethod
    def _add_alignment_metrics(output: TrajectoryBatch) -> None:
        """Expose alignment health implied by the ``TrajectoryBatch`` contract.

        A runner that returns rollout logprobs promises they are position-aligned
        with its response IDs. That direct token-in/token-out path is exact by
        construction. Runners that reconstruct token streams can publish richer
        exact/LCS/failure metrics themselves; those observations take precedence.
        """
        rollout_logprobs = output.get("rollout_logprobs")
        if rollout_logprobs is None:
            return

        rollout_metrics = output.get("rollout_metrics") or {}
        if TIS_ALIGNED_TOKENS_METRIC in rollout_metrics:
            return

        response_ids = output["response_ids"]
        loss_masks = output["loss_masks"]
        if not (len(response_ids) == len(loss_masks) == len(rollout_logprobs)):
            raise ValueError("response IDs, loss masks, and rollout logprobs must have the same batch size")

        aligned_tokens = 0
        for sample_response_ids, sample_loss_mask, sample_logprobs in zip(response_ids, loss_masks, rollout_logprobs):
            if not (len(sample_response_ids) == len(sample_loss_mask) == len(sample_logprobs)):
                raise ValueError("rollout logprobs must align one-for-one with response IDs and loss masks")
            aligned_tokens += sum(bool(value) for value in sample_loss_mask)

        rollout_metrics.update(
            {
                TIS_ALIGNED_TOKENS_METRIC: float(aligned_tokens),
                TIS_EXACT_MATCH_FRACTION_METRIC: 1.0 if aligned_tokens else 0.0,
                TIS_LCS_FALLBACK_FRACTION_METRIC: 0.0,
                TIS_UNALIGNED_FRACTION_METRIC: 0.0,
                TIS_ALIGNMENT_FAIL_COUNT_METRIC: 0.0,
                TIS_LCS_FALLBACK_MESSAGES_METRIC: 0.0,
                TIS_LCS_FALLBACK_ALERT_METRIC: 0.0,
            }
        )
        output["rollout_metrics"] = rollout_metrics

    async def startup(self) -> None:
        """Initialize async resources before training begins.

        Called once after __init__ but before the first run() call.
        Override to initialize resources like orchestrators, connections, etc.

        Default implementation does nothing (for backwards compatibility).
        """
        pass

    async def shutdown(self) -> None:
        """Cleanup async resources after training ends.

        Called once after the last run() call.
        Override to cleanup resources like orchestrators, connections, etc.
        Should be idempotent (safe to call multiple times).

        Default implementation does nothing (for backwards compatibility).
        """
        pass
