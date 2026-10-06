from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Dict, List, Literal, NotRequired, Optional, Sequence, TypedDict, Union

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
    rollout_routed_experts: Optional[List[np.ndarray]]
    teacher_evidence: Optional[TeacherEvidenceBatch]
    distillation: Optional[PreparedTeacherInput]
    token_level_shaping: Optional[List[List[float]]]
    response_span_tags: Optional[List[List[int]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    teacher_route_keys: Optional[List[str]]
    is_last_step: Optional[List[bool]]
    exclude_from_baseline: Optional[List[bool]]


@dataclass(frozen=True)
class BatchFields:
    """Whole-batch channel presence and geometry used by every row slice."""

    present: frozenset[str]
    first_group_list_fields: frozenset[str]
    route_geometry: tuple[int, int, np.dtype] | None
    token_rewards: bool

    def without_routes(self) -> "BatchFields":
        return replace(
            self,
            present=self.present - {"rollout_routed_experts"},
            first_group_list_fields=self.first_group_list_fields - {"rollout_routed_experts"},
            route_geometry=None,
        )

    @classmethod
    def from_batch(cls, batch: TrajectoryBatch) -> "BatchFields":
        """Describe field presence and the first captured route geometry of one group."""
        geometry = next(
            (
                (*row.shape[1:], row.dtype)
                for row in batch.get("rollout_routed_experts") or ()
                if row is not None and row.ndim == 3
            ),
            None,
        )
        return cls(
            present=frozenset(key for key, value in batch.items() if value is not None),
            first_group_list_fields=frozenset(key for key, value in batch.items() if isinstance(value, list)),
            route_geometry=geometry,
            token_rewards=any(isinstance(reward, list) for reward in batch["rewards"]),
        )

    @classmethod
    def from_groups(cls, groups: Sequence["BatchFields"]) -> "BatchFields":
        """Combine ordered group descriptors using the first group's list-field policy."""
        present = frozenset(key for group in groups for key in group.present)
        if "unshaped_rewards" in present and any("unshaped_rewards" not in group.present for group in groups):
            present = present | {"unshaped_reward_available"}
        return cls(
            present=present,
            first_group_list_fields=groups[0].first_group_list_fields,
            route_geometry=next((group.route_geometry for group in groups if group.route_geometry is not None), None),
            token_rewards=any(group.token_rewards for group in groups),
        )


class TitoFullDeclineReason(StrEnum):
    """Reason exact full-token trajectory assembly could not be proven safe."""

    MISSING_STREAMS = "missing_streams"
    EMPTY_STREAMS = "empty_streams"
    TURN_COUNT_MISMATCH = "turn_count_mismatch"
    ASSISTANT_MESSAGE_COUNT_MISMATCH = "assistant_message_count_mismatch"
    MALFORMED_TURN_STREAM = "malformed_turn_stream"
    PREFIX_MISMATCH = "prefix_mismatch"
    INITIAL_PROMPT_TOO_SHORT = "initial_prompt_too_short"
    GENERATION_PROMPT_MISMATCH = "generation_prompt_mismatch"
    COMPLETION_REGION_MISMATCH = "completion_region_mismatch"


@dataclass(frozen=True)
class ShapingObservations:
    """Ordered trajectory scalars and charged credits used by shaping metrics."""

    components: tuple[RewardShapingComponents, ...]
    shaped_totals: tuple[float, ...]
    outcomes: tuple[float, ...]
    response_tokens: tuple[int, ...]
    stop_reasons: tuple[str | None, ...]
    loop_incidence: tuple[bool, ...]
    loop_advantage_totals: tuple[float, ...]
    loop_charged_tokens: tuple[int, ...]
    charged_loop_advantages: tuple[float, ...]


@dataclass(frozen=True)
class RolloutObservations:
    """Ordered rollout scalars, verifier evidence and complete group metric maps."""

    response_lengths: tuple[int, ...]
    optimization_totals: tuple[float, ...]
    reward_sign_totals: tuple[float, ...]
    outcomes: tuple[float, ...]
    unshaped_outcomes: tuple[float, ...] | None
    scalar_rewards: tuple[float, ...] | None
    is_last_step: tuple[bool, ...] | None
    verification_results: tuple[VerificationResult | None, ...] | None
    data_sources: tuple[str | None, ...] | None
    env_metrics: tuple[dict[str, Any], ...] | None
    env_classes: tuple[str, ...] | None
    group_metrics: tuple[dict[str, float], ...]
    shaping: ShapingObservations | None
