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
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
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

    @abstractmethod
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        raise NotImplementedError()

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
