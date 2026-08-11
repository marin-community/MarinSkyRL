from abc import ABC, abstractmethod
from types import MappingProxyType
from skyrl_train.generators.generator_types import (
    BatchMetadata,
    ConversationType,
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
    TrainingPhase,
)
from skyrl_train.generators.trajectory_reward_shaping import shape_trajectory_rewards
from skyrl_train.generators.trajectory_retention import TrajectorySink, retain_trajectories


__all__ = [
    "BatchMetadata",
    "ConversationType",
    "GeneratorInput",
    "GeneratorInterface",
    "GeneratorOutput",
    "TrajectoryID",
    "TrainingPhase",
]


class GeneratorInterface(ABC):
    """Abstract base class for trajectory generators.

    Lifecycle:
        1. __init__() - Synchronous initialization (no async resources)
        2. startup() - Async initialization of resources (e.g., orchestrators, connections)
        3. generate() - Called repeatedly during training
        4. shutdown() - Async cleanup of resources

    Implementations should handle errors gracefully in generate() to avoid killing the
    training job. Use restart logic for recoverable failures.
    """

    generator_cfg = MappingProxyType({})
    trajectory_sink: TrajectorySink | None = None

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Generate trajectories and apply generator-independent output finalization.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        output = await self._generate(input_batch, disable_tqdm=disable_tqdm)
        shape_trajectory_rewards(output, self.generator_cfg.get("trajectory_reward_shaping"))
        self._add_alignment_metrics(output)
        if self.trajectory_sink is not None:
            await retain_trajectories(self.trajectory_sink, input_batch, output)
        return output

    def set_trajectory_sink(self, sink: TrajectorySink) -> None:
        """Attach the trainer-owned sink used by shared output finalization."""
        sink.bind_generator(type(self).__name__)
        self.trajectory_sink = sink

    @abstractmethod
    async def _generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Produce trajectories before shared output finalization."""
        raise NotImplementedError()

    @staticmethod
    def _add_alignment_metrics(output: GeneratorOutput) -> None:
        """Expose alignment health implied by the ``GeneratorOutput`` contract.

        A generator that returns rollout logprobs promises they are position-aligned
        with its response IDs. That direct token-in/token-out path is exact by
        construction. Generators that reconstruct token streams can publish richer
        exact/LCS/failure metrics themselves; those observations take precedence.
        """
        rollout_logprobs = output.get("rollout_logprobs")
        if rollout_logprobs is None:
            return

        rollout_metrics = output.get("rollout_metrics") or {}
        if "generate/tis/aligned_tokens" in rollout_metrics:
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
                "generate/tis/aligned_tokens": float(aligned_tokens),
                "generate/tis/exact_match_fraction": 1.0 if aligned_tokens else 0.0,
                "generate/tis/lcs_fallback_fraction": 0.0,
                "generate/tis/unaligned_fraction": 0.0,
                "generate/tis/alignment_fail_count": 0.0,
                "generate/tis/lcs_fallback_messages": 0.0,
                "generate/tis/lcs_fallback_alert": 0.0,
            }
        )
        output["rollout_metrics"] = rollout_metrics

    async def startup(self) -> None:
        """Initialize async resources before training begins.

        Called once after __init__ but before the first generate() call.
        Override to initialize resources like orchestrators, connections, etc.

        Default implementation does nothing (for backwards compatibility).
        """
        pass

    async def shutdown(self) -> None:
        """Cleanup async resources after training ends.

        Called once after the last generate() call.
        Override to cleanup resources like orchestrators, connections, etc.
        Should be idempotent (safe to call multiple times).

        Default implementation does nothing (for backwards compatibility).
        """
        pass
