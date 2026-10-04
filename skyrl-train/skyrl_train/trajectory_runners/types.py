from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union

import numpy as np

from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult
from skyrl_train.distillation import PreparedTeacherInput, TeacherEvidenceBatch
from skyrl_train.inference_engines.base import ConversationType


TrainingPhase = Literal["train", "eval"]
REWARD_SHAPING_COMPONENT_NAMES = ("passthrough", "non_termination", "overlong", "successful_length")


class TokenProvenance(StrEnum):
    """How response token IDs were obtained from a model transport."""

    ENGINE = "engine"
    RECONSTRUCTED = "reconstructed"


@dataclass
class AgentLoopOutput:
    """One normalized SkyRL-Gym interaction result."""

    evidence: RolloutEvidence
    verification: VerificationResult
    reward: RewardResult
    disposition: TrainingDisposition
    loss_mask: List[int]
    env_metrics: Dict[str, Any]
    token_provenance: TokenProvenance = TokenProvenance.ENGINE
    error_treatment: Optional[str] = None


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


class TrajectoryRequestBatch(TypedDict):
    prompts: List[ConversationType]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[TrajectoryID]]
    batch_metadata: Optional[BatchMetadata]


class RewardShapingComponents(TypedDict):
    passthrough: float
    non_termination: float
    overlong: float
    successful_length: float


class RewardShapingLoopSpan(TypedDict):
    start: int
    end: int


class VerifierTestRecord(TypedDict):
    record_id: str
    trial_id: TrajectoryID
    test_id: str
    outcome: str
    output: str


class VerifierTestCollection(TypedDict):
    parser: Optional[str]
    complete: bool
    tests: List[VerifierTestRecord]


class TrajectoryBatch(TypedDict):
    """Normalized output shared by trajectory runners and trainer consumers.

    Raw outcomes remain separate from optimization rewards. Optional diagnostic
    channels are absent unless their corresponding feature is active.
    ``env_metrics`` and ``env_classes`` are present together or both absent;
    when present, each has one entry per trajectory row.
    """

    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    data_sources: Optional[List[str | None]]
    rewards: Union[List[float], List[List[float]]]
    unshaped_rewards: Optional[List[float]]
    unshaped_reward_available: Optional[List[bool]]
    reward_shaping_components: Optional[List[RewardShapingComponents]]
    reward_shaping_loop_spans: Optional[List[List[RewardShapingLoopSpan]]]
    loop_advantages: Optional[List[List[float]]]
    reward_shaping_versions: Optional[List[int]]
    verification_results: List[Optional[VerificationResult]]
    evidence_messages: List[Optional[list[dict[str, Any]]]]
    verifier_tests: Optional[List[Optional[VerifierTestCollection]]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    exception_types: Optional[List[Optional[str]]]
    error_treatments: Optional[List[Optional[str]]]
    server_errors: Optional[List[Optional[Dict[str, Any]]]]
    rollout_metrics: Optional[Dict[str, Any]]
    env_metrics: NotRequired[List[Dict[str, Any]]]
    env_classes: NotRequired[List[str]]
    rollout_logprobs: Optional[List[np.ndarray]]
    student_topk_indices: Optional[List[np.ndarray]]
    behavior_topk_logprobs: Optional[List[np.ndarray]]
    token_policy_versions: NotRequired[List[List[int]]]
    rollout_routed_experts: Optional[List[np.ndarray]]
    teacher_evidence: Optional[TeacherEvidenceBatch]
    distillation: Optional[PreparedTeacherInput]
    token_level_shaping: Optional[List[List[float]]]
    response_span_tags: Optional[List[List[int]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    teacher_route_keys: Optional[List[str]]
    is_last_step: Optional[List[bool]]
    exclude_from_baseline: Optional[List[bool]]
