from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from skyrl_train.inference_engines.base import ConversationType


TrainingPhase = Literal["train", "eval"]


@dataclass
class TrajectoryID:
    instance_id: str
    repetition_id: int

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


class RewardShapingComponents(TypedDict):
    non_termination: float
    successful_length: float


class RewardShapingLoopSpan(TypedDict):
    start: int
    end: int


class GeneratorOutput(TypedDict):
    """Normalized output shared by trajectory generators and trainer consumers.

    Raw outcomes remain separate from optimization rewards. Optional diagnostic
    channels are absent unless their corresponding feature is active.
    """

    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    unshaped_rewards: Optional[List[float]]
    reward_shaping_components: Optional[List[RewardShapingComponents]]
    reward_shaping_loop_spans: Optional[List[List[RewardShapingLoopSpan]]]
    loop_advantages: Optional[List[List[float]]]
    reward_shaping_versions: Optional[List[int]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    rollout_logprobs: Optional[List[List[float]]]
    rollout_routed_experts: Optional[List[List[List[List[int]]]]]
    token_level_shaping: Optional[List[List[float]]]
    response_span_tags: Optional[List[List[int]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    is_last_step: Optional[List[bool]]
    exclude_from_baseline: Optional[List[bool]]
    actual_global_step: Optional[int]
