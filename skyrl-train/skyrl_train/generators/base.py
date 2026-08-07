from typing import List, Dict, Any, TypedDict, Optional, Union, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from skyrl_train.inference_engines.base import ConversationType

TrainingPhase = Literal["train", "eval"]


@dataclass
class TrajectoryID:
    instance_id: str  # Unique identifier for the instance in the dataset
    repetition_id: int  # Which sample/repetition for this UID (0, 1, 2... for GRPO)

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"


@dataclass
class BatchMetadata:
    global_step: int
    training_phase: TrainingPhase


class GeneratorInput(TypedDict):
    prompts: List[ConversationType]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[TrajectoryID]]
    batch_metadata: Optional[BatchMetadata]


class GeneratorOutput(TypedDict):
    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    # Outcome rewards before optimization-specific shaping. Metrics such as
    # pass@k use this channel so changing a shaper cannot change task success.
    # Generators without a distinct shaping stage may omit it.
    unshaped_rewards: Optional[List[float]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    # Sampling-policy logprobs aligned one-for-one with response_ids. Generation
    # strategies populate this channel whenever sampling requested logprobs,
    # independent of whether requests were batched. If token identity is lost
    # (for example through text postprocessing), return None for the batch rather
    # than attach probabilities to different tokens.
    rollout_logprobs: Optional[List[List[float]]]
    # MoE router-replay (Stage 1 capture rail): per-sample per-token [L, K]
    # expert-selection rows, i.e. List[ [response_len, L, K] ]. Present only when
    # trainer.policy.fsdp_config.moe_router_replay is True; absent otherwise so the
    # default (production) GeneratorOutput is byte-identical.
    rollout_routed_experts: Optional[List[List[List[List[int]]]]]
    # Loop-behavior reward shaping (Stage B / F5): per-sample per-token additive
    # shaping channel, List[ [response_len] float ]. Present only when
    # enable_token_reward_channel is True; absent otherwise so the default
    # GeneratorOutput is byte-identical. Carries ZEROS in Stage B (no-op).
    token_level_shaping: Optional[List[List[float]]]
    # Loop-behavior reward shaping (Stage B / F4): per-sample per-token span tags,
    # List[ [response_len] int ] ({OTHER=0,THINK=1,ACTION=2,EDIT=3}). Present only
    # when enable_token_reward_channel is True.
    response_span_tags: Optional[List[List[int]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    # Applicable only for step-wise training
    is_last_step: Optional[List[bool]]
    # For RLOO-N: exclude sample from baseline computation (e.g., infrastructure failures)
    # When True, the sample is masked from loss AND excluded from group baseline calculation.
    # This allows distinguishing infrastructure failures (exclude) from agent failures (include with zero reward).
    exclude_from_baseline: Optional[List[bool]]
    # Actual global_step captured at first vLLM inference (for accurate staleness tracking).
    # Scalar — same for all samples in a group since they share one generation episode.
    actual_global_step: Optional[int]


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

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        """Generate trajectories and apply generator-independent output finalization.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        output = await self._generate(input_batch, disable_tqdm=disable_tqdm)
        self._add_alignment_metrics(output)
        return output

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
