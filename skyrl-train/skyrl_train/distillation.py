"""Teacher evidence contracts and backend-neutral distillation objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol, Self

import torch
from omegaconf import DictConfig

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.tensor_math import TOKEN_MEAN_LOSS_REDUCTION, masked_mean, safe_exp_delta

INVALID_TOPK_INDEX = -1
RETAINED_MASS_ATOL = 1e-6
DISTILLATION_SCORED_TOKENS_METRIC = "distillation/scored_tokens"
DISTILLATION_TOPK_METRIC = "distillation_topk"


@dataclass(frozen=True)
class TeacherScoreRequest:
    """Exact student actions submitted to one logical teacher."""

    trajectory_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    teacher_id: str
    tokenizer_fingerprint: str
    plan_version: str
    prompt_token_ids: torch.Tensor
    prompt_mask: torch.Tensor
    response_token_ids: torch.Tensor
    response_mask: torch.Tensor
    evidence: TeacherEvidenceKind
    top_k: Optional[int] = None
    student_topk_indices: Optional[torch.Tensor] = None
    behavior_topk_logprobs: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TeacherEvidence:
    """Provenance common to every teacher evidence representation."""

    trajectory_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    teacher_id: str
    teacher_revision: str
    plan_version: str
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class ChosenTokenTeacherEvidence(TeacherEvidence):
    """Teacher probability of each emitted student token."""

    chosen_logprobs: torch.Tensor
    kind: ClassVar[TeacherEvidenceKind] = TeacherEvidenceKind.CHOSEN_TOKEN


@dataclass(frozen=True)
class TopKTeacherEvidence(TeacherEvidence):
    """Sparse teacher distributions with explicit retained probability mass."""

    topk_indices: torch.Tensor
    topk_logprobs: torch.Tensor
    retained_mass: torch.Tensor
    kind: ClassVar[TeacherEvidenceKind] = TeacherEvidenceKind.TOPK_DISTRIBUTION


@dataclass(frozen=True)
class StudentSelectedTeacherEvidence(TeacherEvidence):
    """Teacher scores on the exact student-selected token IDs in the request."""

    student_topk_indices: torch.Tensor
    teacher_on_student_logprobs: torch.Tensor
    kind: ClassVar[TeacherEvidenceKind] = TeacherEvidenceKind.STUDENT_SELECTED_TOPK


TeacherEvidenceBatch = ChosenTokenTeacherEvidence | TopKTeacherEvidence | StudentSelectedTeacherEvidence


@dataclass(frozen=True)
class SampledReverseKLInput:
    """Minimal learner payload for sampled chosen-token reverse KL."""

    teacher_action_log_probs: torch.Tensor
    valid_mask: torch.Tensor
    loss_weights: torch.Tensor

    def to(self, device: torch.device) -> Self:
        return type(self)(
            teacher_action_log_probs=self.teacher_action_log_probs.to(device),
            valid_mask=self.valid_mask.to(device),
            loss_weights=self.loss_weights.to(device),
        )

    def pin_memory(self) -> Self:
        return type(self)(
            teacher_action_log_probs=self.teacher_action_log_probs.pin_memory(),
            valid_mask=self.valid_mask.pin_memory(),
            loss_weights=self.loss_weights.pin_memory(),
        )

    def student_token_ids(self) -> None:
        return None

    def training_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "teacher_action_log_probs": self.teacher_action_log_probs,
            "teacher_valid_mask": self.valid_mask,
            "distillation_loss_weights": self.loss_weights,
        }

    def objective_loss(
        self,
        *,
        action_log_probs: torch.Tensor,
        old_action_log_probs: torch.Tensor,
        student_selected_logprobs: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        config: DictConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del config
        if student_selected_logprobs is not None:
            raise ValueError("chosen-token reverse KL does not use selected student logprobs")
        return sampled_reverse_kl_loss(action_log_probs, old_action_log_probs, self, loss_mask), {}


@dataclass(frozen=True)
class SparseForwardKLInput:
    """Minimal learner payload for retained-mass-normalized top-K forward KL."""

    teacher_topk_indices: torch.Tensor
    teacher_topk_logprobs: torch.Tensor
    retained_mass: torch.Tensor
    valid_mask: torch.Tensor
    loss_weights: torch.Tensor

    def to(self, device: torch.device) -> Self:
        return type(self)(
            teacher_topk_indices=self.teacher_topk_indices.to(device),
            teacher_topk_logprobs=self.teacher_topk_logprobs.to(device),
            retained_mass=self.retained_mass.to(device),
            valid_mask=self.valid_mask.to(device),
            loss_weights=self.loss_weights.to(device),
        )

    def pin_memory(self) -> Self:
        return type(self)(
            teacher_topk_indices=self.teacher_topk_indices.pin_memory(),
            teacher_topk_logprobs=self.teacher_topk_logprobs.pin_memory(),
            retained_mass=self.retained_mass.pin_memory(),
            valid_mask=self.valid_mask.pin_memory(),
            loss_weights=self.loss_weights.pin_memory(),
        )

    def student_token_ids(self) -> torch.Tensor:
        return self.teacher_topk_indices

    def training_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "teacher_topk_indices": self.teacher_topk_indices,
            "teacher_topk_logprobs": self.teacher_topk_logprobs,
            "teacher_retained_mass": self.retained_mass,
            "teacher_valid_mask": self.valid_mask,
            "distillation_loss_weights": self.loss_weights,
        }

    def objective_loss(
        self,
        *,
        action_log_probs: torch.Tensor,
        old_action_log_probs: torch.Tensor,
        student_selected_logprobs: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        config: DictConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del action_log_probs, old_action_log_probs, config
        if student_selected_logprobs is None:
            raise ValueError("sparse forward KL requires student logprobs at the teacher's top-K token IDs")
        return sparse_forward_kl_loss(student_selected_logprobs, self, loss_mask)


@dataclass(frozen=True)
class StudentTopKPolicySurrogateInput:
    """Learner payload for the released student-selected top-K OPD surrogate."""

    student_topk_indices: torch.Tensor
    behavior_topk_logprobs: torch.Tensor
    teacher_on_student_logprobs: torch.Tensor
    valid_mask: torch.Tensor
    loss_weights: torch.Tensor

    def to(self, device: torch.device) -> Self:
        return type(self)(
            student_topk_indices=self.student_topk_indices.to(device),
            behavior_topk_logprobs=self.behavior_topk_logprobs.to(device),
            teacher_on_student_logprobs=self.teacher_on_student_logprobs.to(device),
            valid_mask=self.valid_mask.to(device),
            loss_weights=self.loss_weights.to(device),
        )

    def pin_memory(self) -> Self:
        return type(self)(
            student_topk_indices=self.student_topk_indices.pin_memory(),
            behavior_topk_logprobs=self.behavior_topk_logprobs.pin_memory(),
            teacher_on_student_logprobs=self.teacher_on_student_logprobs.pin_memory(),
            valid_mask=self.valid_mask.pin_memory(),
            loss_weights=self.loss_weights.pin_memory(),
        )

    def student_token_ids(self) -> torch.Tensor:
        return self.student_topk_indices

    def training_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "student_topk_indices": self.student_topk_indices,
            "behavior_topk_logprobs": self.behavior_topk_logprobs,
            "teacher_on_student_logprobs": self.teacher_on_student_logprobs,
            "teacher_valid_mask": self.valid_mask,
            "distillation_loss_weights": self.loss_weights,
        }

    def objective_loss(
        self,
        *,
        action_log_probs: torch.Tensor,
        old_action_log_probs: torch.Tensor,
        student_selected_logprobs: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        config: DictConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del action_log_probs, old_action_log_probs
        if student_selected_logprobs is None:
            raise ValueError("student-top-K OPD requires student logprobs at the selected IDs")
        return student_topk_policy_surrogate_loss(student_selected_logprobs, self, loss_mask, config)


class DistillationInput(Protocol):
    valid_mask: torch.Tensor
    loss_weights: torch.Tensor

    def to(self, device: torch.device) -> Self: ...

    def pin_memory(self) -> Self: ...

    def student_token_ids(self) -> Optional[torch.Tensor]: ...

    def training_tensors(self) -> dict[str, torch.Tensor]: ...

    def objective_loss(
        self,
        *,
        action_log_probs: torch.Tensor,
        old_action_log_probs: torch.Tensor,
        student_selected_logprobs: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        config: DictConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]: ...


def distillation_input_from_tensors(
    *,
    teacher_action_log_probs: Optional[torch.Tensor],
    teacher_topk_indices: Optional[torch.Tensor],
    teacher_topk_logprobs: Optional[torch.Tensor],
    teacher_retained_mass: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    loss_weights: Optional[torch.Tensor],
    student_topk_indices: Optional[torch.Tensor] = None,
    behavior_topk_logprobs: Optional[torch.Tensor] = None,
    teacher_on_student_logprobs: Optional[torch.Tensor] = None,
) -> Optional[DistillationInput]:
    """Build exactly one learner payload and reject partial or mixed variants."""
    chosen_present = teacher_action_log_probs is not None
    topk_values = (teacher_topk_indices, teacher_topk_logprobs, teacher_retained_mass)
    topk_present = any(value is not None for value in topk_values)
    student_topk_values = (student_topk_indices, behavior_topk_logprobs, teacher_on_student_logprobs)
    student_topk_present = any(value is not None for value in student_topk_values)
    if not chosen_present and not topk_present and not student_topk_present:
        if valid_mask is not None or loss_weights is not None:
            raise ValueError("distillation masks and weights require teacher evidence")
        return None
    if sum((chosen_present, topk_present, student_topk_present)) != 1:
        raise ValueError("distillation cannot mix objective evidence variants")
    if valid_mask is None or loss_weights is None:
        raise ValueError("distillation evidence requires valid_mask and loss_weights")
    if chosen_present:
        return SampledReverseKLInput(teacher_action_log_probs, valid_mask, loss_weights)
    if student_topk_present:
        if any(value is None for value in student_topk_values):
            raise ValueError("student-top-K OPD requires selected IDs, behavior logprobs, and teacher scores together")
        assert student_topk_indices is not None
        assert behavior_topk_logprobs is not None
        assert teacher_on_student_logprobs is not None
        return StudentTopKPolicySurrogateInput(
            student_topk_indices, behavior_topk_logprobs, teacher_on_student_logprobs, valid_mask, loss_weights
        )
    if any(value is None for value in topk_values):
        raise ValueError("sparse forward KL requires top-K indices, logprobs, and retained_mass together")
    assert teacher_topk_indices is not None
    assert teacher_topk_logprobs is not None
    assert teacher_retained_mass is not None
    return SparseForwardKLInput(
        teacher_topk_indices=teacher_topk_indices,
        teacher_topk_logprobs=teacher_topk_logprobs,
        retained_mass=teacher_retained_mass,
        valid_mask=valid_mask,
        loss_weights=loss_weights,
    )


def _require_nonempty(value: str, path: str) -> None:
    if not value.strip():
        raise ValueError(f"{path} must be a non-empty string")


def _validate_masked_logprobs(logprobs: torch.Tensor, valid_mask: torch.Tensor, label: str) -> None:
    if not torch.is_floating_point(logprobs):
        raise ValueError(f"{label} must have floating-point dtype")
    valid_values = logprobs[valid_mask]
    if not torch.all(torch.isfinite(valid_values)) or torch.any(valid_values > 0):
        raise ValueError(f"valid {label} must be finite and no greater than zero")
    if not torch.all(torch.isnan(logprobs[~valid_mask])):
        raise ValueError(f"invalid {label} must be NaN, not plausible scores")


def _validate_loss_weights(loss_weights: torch.Tensor, label: str) -> None:
    if not torch.is_floating_point(loss_weights):
        raise ValueError(f"{label} must have floating-point dtype")
    if not torch.all(torch.isfinite(loss_weights)) or torch.any(loss_weights < 0):
        raise ValueError(f"{label} must be finite and non-negative")


def _validate_selected_token_ids(indices: torch.Tensor, valid_mask: torch.Tensor, label: str) -> None:
    valid_indices = indices[valid_mask]
    if torch.any(valid_indices < 0):
        raise ValueError(f"valid {label} indices must be non-negative")
    if not torch.all(indices[~valid_mask] == INVALID_TOPK_INDEX):
        raise ValueError(f"invalid {label} indices must use the {INVALID_TOPK_INDEX} sentinel")
    if valid_indices.numel():
        sorted_indices = valid_indices.sort(dim=-1).values
        if torch.any(sorted_indices[..., 1:] == sorted_indices[..., :-1]):
            raise ValueError(f"valid {label} indices must be unique per token")


def _validate_topk_distribution(
    indices: torch.Tensor,
    logprobs: torch.Tensor,
    retained_mass: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if indices.shape != logprobs.shape or indices.ndim != 3:
        raise ValueError("teacher top-K indices and logprobs must have matching [batch, response_len, K] shapes")
    if indices.shape[2] == 0:
        raise ValueError("teacher top-K evidence must retain at least one token per valid position")
    if indices.shape[:2] != valid_mask.shape:
        raise ValueError("teacher top-K evidence must match valid_mask response coordinates")
    if retained_mass.shape != valid_mask.shape:
        raise ValueError("teacher retained_mass must match valid_mask")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher top-K indices must have integer dtype")
    _validate_masked_logprobs(logprobs, valid_mask, "teacher top-K logprobs")
    if not torch.is_floating_point(retained_mass):
        raise ValueError("teacher retained_mass must have floating-point dtype")
    valid_logprobs = logprobs[valid_mask]
    valid_mass = retained_mass[valid_mask]
    _validate_selected_token_ids(indices, valid_mask, "teacher top-K")
    if not torch.all(torch.isfinite(valid_mass)) or torch.any(
        (valid_mass <= 0) | (valid_mass > 1 + RETAINED_MASS_ATOL)
    ):
        raise ValueError("valid teacher retained_mass must lie in (0, 1]")
    if not torch.all(torch.isnan(retained_mass[~valid_mask])):
        raise ValueError("invalid teacher retained_mass must be NaN")
    observed_mass = valid_logprobs.float().exp().sum(dim=-1)
    if not torch.allclose(observed_mass, valid_mass.float(), rtol=1e-4, atol=RETAINED_MASS_ATOL):
        raise ValueError("teacher retained_mass must equal the probability mass of top-K logprobs")


def _validate_evidence_coordinates(
    evidence: TeacherEvidenceBatch,
    *,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
    route_ids: Optional[tuple[str, ...]] = None,
) -> None:
    if evidence.trajectory_ids != trajectory_ids:
        raise ValueError("teacher evidence trajectory_ids do not match the requested trajectories")
    if route_ids is not None:
        if evidence.route_ids != route_ids:
            raise ValueError("teacher evidence route_ids do not match the requested routes")
    elif len(evidence.route_ids) != len(trajectory_ids):
        raise ValueError("teacher evidence route_ids do not align with the requested trajectories")
    if response_mask.dtype is not torch.bool:
        raise ValueError("response_mask must have bool dtype")
    if evidence.valid_mask.shape != response_mask.shape:
        raise ValueError("teacher evidence valid_mask must match the requested response coordinates")
    if evidence.valid_mask.dtype is not torch.bool:
        raise ValueError("teacher evidence valid_mask must have bool dtype")
    if torch.any(evidence.valid_mask & ~response_mask):
        raise ValueError("teacher evidence cannot mark padded response positions valid")


def validate_teacher_score_request(request: TeacherScoreRequest) -> None:
    """Reject requests with ambiguous identities or token coordinates."""
    batch_size = len(request.trajectory_ids)
    if batch_size == 0:
        raise ValueError("teacher score request must contain at least one trajectory")
    if len(set(request.trajectory_ids)) != batch_size:
        raise ValueError("teacher score request trajectory_ids must be unique")
    if len(request.route_ids) != batch_size:
        raise ValueError("teacher score request route_ids must align with trajectory_ids")
    for trajectory_id in request.trajectory_ids:
        _require_nonempty(trajectory_id, "teacher score request trajectory_id")
    for route_id in request.route_ids:
        _require_nonempty(route_id, "teacher score request route_id")
    _require_nonempty(request.teacher_id, "teacher score request teacher_id")
    _require_nonempty(request.tokenizer_fingerprint, "teacher score request tokenizer_fingerprint")
    _require_nonempty(request.plan_version, "teacher score request plan_version")
    if request.prompt_token_ids.ndim != 2:
        raise ValueError("teacher score request prompt_token_ids must have shape [batch, prompt_len]")
    if request.prompt_mask.shape != request.prompt_token_ids.shape:
        raise ValueError("teacher score request prompt_mask must match prompt_token_ids")
    if request.prompt_mask.dtype is not torch.bool:
        raise ValueError("teacher score request prompt_mask must have bool dtype")
    if request.response_token_ids.ndim != 2:
        raise ValueError("teacher score request response_token_ids must have shape [batch, response_len]")
    if request.response_mask.shape != request.response_token_ids.shape:
        raise ValueError("teacher score request response_mask must match response_token_ids")
    if request.response_mask.dtype is not torch.bool:
        raise ValueError("teacher score request response_mask must have bool dtype")
    if request.prompt_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher score request prompt_token_ids must have integer dtype")
    if request.response_token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher score request response_token_ids must have integer dtype")
    if torch.any(request.prompt_token_ids < 0) or torch.any(request.response_token_ids < 0):
        raise ValueError("teacher score request token IDs must be non-negative")
    if request.prompt_token_ids.shape[0] != batch_size or request.response_token_ids.shape[0] != batch_size:
        raise ValueError("teacher score request tensors must align with trajectory_ids")
    if request.evidence is TeacherEvidenceKind.STUDENT_SELECTED_TOPK:
        if request.student_topk_indices is None or request.behavior_topk_logprobs is None:
            raise ValueError("student-selected teacher requests require indices and behavior logprobs")
        if (
            request.student_topk_indices.ndim != 3
            or request.student_topk_indices.shape[:2] != request.response_mask.shape
        ):
            raise ValueError("student-selected indices must have [batch, response_len, K] coordinates")
        if request.student_topk_indices.shape[-1] == 0:
            raise ValueError("student-selected requests require at least one token per valid position")
        if request.behavior_topk_logprobs.shape != request.student_topk_indices.shape:
            raise ValueError("student-selected behavior logprobs must match selected indices")
        if isinstance(request.top_k, bool) or request.top_k != request.student_topk_indices.shape[-1]:
            raise ValueError("student-selected top_k must match selected indices")
        if request.student_topk_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("student-selected indices must have integer dtype")
        _validate_selected_token_ids(request.student_topk_indices, request.response_mask, "student-selected")
        _validate_masked_logprobs(request.behavior_topk_logprobs, request.response_mask, "behavior top-K logprobs")
    elif request.student_topk_indices is not None or request.behavior_topk_logprobs is not None:
        raise ValueError("student-selected fields require student-selected teacher evidence")
    if request.evidence is TeacherEvidenceKind.TOPK_DISTRIBUTION:
        if isinstance(request.top_k, bool) or not isinstance(request.top_k, int) or request.top_k <= 0:
            raise ValueError("top-K teacher score requests require a positive top_k")
    elif request.evidence is not TeacherEvidenceKind.STUDENT_SELECTED_TOPK and request.top_k is not None:
        raise ValueError("top_k is only valid for top-K teacher score requests")


def validate_teacher_evidence(request: TeacherScoreRequest, evidence: TeacherEvidenceBatch) -> None:
    """Reject evidence whose identity, token coordinates, shape, or values are ambiguous."""
    validate_teacher_score_request(request)
    if not isinstance(evidence, (ChosenTokenTeacherEvidence, TopKTeacherEvidence, StudentSelectedTeacherEvidence)):
        raise TypeError(f"unsupported teacher evidence type: {type(evidence).__name__}")
    _validate_evidence_coordinates(
        evidence,
        trajectory_ids=request.trajectory_ids,
        route_ids=request.route_ids,
        response_mask=request.response_mask,
    )
    if evidence.teacher_id != request.teacher_id:
        raise ValueError("teacher evidence teacher_id does not match the score request")
    if evidence.kind is not request.evidence:
        raise ValueError("teacher evidence kind does not match the score request")
    _require_nonempty(evidence.teacher_revision, "teacher evidence teacher_revision")
    _require_nonempty(evidence.plan_version, "teacher evidence plan_version")
    if evidence.plan_version != request.plan_version:
        raise ValueError("teacher evidence plan_version does not match the score request")

    if isinstance(evidence, ChosenTokenTeacherEvidence):
        if evidence.chosen_logprobs.shape != evidence.valid_mask.shape:
            raise ValueError("teacher chosen_logprobs must match valid_mask")
        _validate_masked_logprobs(evidence.chosen_logprobs, evidence.valid_mask, "teacher chosen_logprobs")
        return

    if isinstance(evidence, StudentSelectedTeacherEvidence):
        assert request.student_topk_indices is not None
        if not torch.equal(evidence.student_topk_indices, request.student_topk_indices):
            raise ValueError("teacher evidence must identify the exact student-selected token IDs")
        if evidence.teacher_on_student_logprobs.shape != request.student_topk_indices.shape:
            raise ValueError("teacher scores must match student-selected token coordinates")
        _validate_masked_logprobs(
            evidence.teacher_on_student_logprobs, evidence.valid_mask, "teacher-on-student logprobs"
        )
        return

    _validate_topk_distribution(
        evidence.topk_indices,
        evidence.topk_logprobs,
        evidence.retained_mass,
        evidence.valid_mask,
    )
    if evidence.topk_indices.shape[-1] != request.top_k:
        raise ValueError(f"teacher evidence top-K width must match requested top_k={request.top_k}")


def prepare_sampled_reverse_kl(
    request: TeacherScoreRequest,
    evidence: ChosenTokenTeacherEvidence,
    *,
    coefficient: float,
    route_weights: torch.Tensor,
) -> SampledReverseKLInput:
    """Validate chosen-token evidence and compile its minimal learner payload."""
    validate_teacher_evidence(request, evidence)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("distillation coefficient must be a positive finite number")
    if route_weights.shape != evidence.valid_mask.shape:
        raise ValueError("distillation route_weights must match teacher response coordinates")
    _validate_loss_weights(route_weights, "distillation route_weights")
    return SampledReverseKLInput(
        teacher_action_log_probs=evidence.chosen_logprobs,
        valid_mask=evidence.valid_mask,
        loss_weights=route_weights * coefficient,
    )


def prepare_sparse_forward_kl(
    request: TeacherScoreRequest,
    evidence: TopKTeacherEvidence,
    *,
    coefficient: float,
    route_weights: torch.Tensor,
) -> SparseForwardKLInput:
    """Validate top-K evidence and compile its sparse learner payload."""
    validate_teacher_evidence(request, evidence)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("distillation coefficient must be a positive finite number")
    if route_weights.shape != evidence.valid_mask.shape:
        raise ValueError("distillation route_weights must match teacher response coordinates")
    _validate_loss_weights(route_weights, "distillation route_weights")
    return SparseForwardKLInput(
        teacher_topk_indices=evidence.topk_indices,
        teacher_topk_logprobs=evidence.topk_logprobs,
        retained_mass=evidence.retained_mass,
        valid_mask=evidence.valid_mask,
        loss_weights=route_weights * coefficient,
    )


def prepare_student_topk_policy_surrogate(
    request: TeacherScoreRequest,
    evidence: StudentSelectedTeacherEvidence,
    *,
    coefficient: float,
    route_weights: torch.Tensor,
) -> StudentTopKPolicySurrogateInput:
    """Compile exact student-selected evidence into the shared learner payload."""
    validate_teacher_evidence(request, evidence)
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("distillation coefficient must be a positive finite number")
    if route_weights.shape != evidence.valid_mask.shape:
        raise ValueError("distillation route_weights must match teacher response coordinates")
    _validate_loss_weights(route_weights, "distillation route_weights")
    assert request.student_topk_indices is not None
    assert request.behavior_topk_logprobs is not None
    return StudentTopKPolicySurrogateInput(
        student_topk_indices=request.student_topk_indices,
        behavior_topk_logprobs=request.behavior_topk_logprobs,
        teacher_on_student_logprobs=evidence.teacher_on_student_logprobs,
        valid_mask=evidence.valid_mask,
        loss_weights=route_weights * coefficient,
    )


def validate_sampled_reverse_kl_attachment(
    evidence: ChosenTokenTeacherEvidence,
    distillation: SampledReverseKLInput,
    *,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
) -> None:
    """Validate prepared evidence at the trajectory-to-learner boundary."""
    _validate_evidence_coordinates(
        evidence,
        trajectory_ids=trajectory_ids,
        response_mask=response_mask,
    )
    if not torch.equal(distillation.valid_mask, evidence.valid_mask):
        raise ValueError("prepared distillation valid_mask does not match teacher evidence")
    if not torch.allclose(
        distillation.teacher_action_log_probs,
        evidence.chosen_logprobs,
        rtol=0,
        atol=0,
        equal_nan=True,
    ):
        raise ValueError("prepared distillation logprobs do not match teacher evidence")
    if distillation.loss_weights.shape != response_mask.shape:
        raise ValueError("prepared distillation loss_weights must match trajectory response coordinates")
    _validate_loss_weights(distillation.loss_weights, "prepared distillation loss_weights")


def validate_distillation_attachment(
    evidence: TeacherEvidenceBatch,
    distillation: DistillationInput,
    *,
    trajectory_ids: tuple[str, ...],
    response_mask: torch.Tensor,
) -> None:
    """Validate either evidence representation at the trajectory-to-learner boundary."""
    if isinstance(evidence, ChosenTokenTeacherEvidence) and isinstance(distillation, SampledReverseKLInput):
        validate_sampled_reverse_kl_attachment(
            evidence,
            distillation,
            trajectory_ids=trajectory_ids,
            response_mask=response_mask,
        )
        return
    if isinstance(evidence, StudentSelectedTeacherEvidence) and isinstance(
        distillation, StudentTopKPolicySurrogateInput
    ):
        _validate_evidence_coordinates(evidence, trajectory_ids=trajectory_ids, response_mask=response_mask)
        if not torch.equal(distillation.student_topk_indices, evidence.student_topk_indices):
            raise ValueError("prepared student-selected token IDs do not match teacher evidence")
        if distillation.student_topk_indices.shape[:2] != response_mask.shape:
            raise ValueError("prepared student-selected token IDs do not match response coordinates")
        if not torch.equal(distillation.valid_mask, evidence.valid_mask):
            raise ValueError("prepared student-selected valid_mask does not match teacher evidence")
        _validate_selected_token_ids(distillation.student_topk_indices, distillation.valid_mask, "student-selected")
        if distillation.behavior_topk_logprobs.shape != distillation.student_topk_indices.shape:
            raise ValueError("prepared student-selected behavior scores do not match selected token IDs")
        _validate_masked_logprobs(
            distillation.behavior_topk_logprobs, distillation.valid_mask, "prepared behavior top-K logprobs"
        )
        if not torch.allclose(
            distillation.teacher_on_student_logprobs,
            evidence.teacher_on_student_logprobs,
            rtol=0,
            atol=0,
            equal_nan=True,
        ):
            raise ValueError("prepared student-selected teacher scores do not match teacher evidence")
        if distillation.loss_weights.shape != response_mask.shape:
            raise ValueError("prepared distillation loss_weights must match trajectory response coordinates")
        _validate_loss_weights(distillation.loss_weights, "prepared distillation loss_weights")
        return
    if not isinstance(evidence, TopKTeacherEvidence) or not isinstance(distillation, SparseForwardKLInput):
        raise ValueError("teacher evidence and prepared distillation objective kinds must match")
    _validate_evidence_coordinates(
        evidence,
        trajectory_ids=trajectory_ids,
        response_mask=response_mask,
    )
    payload_pairs = (
        ("indices", distillation.teacher_topk_indices, evidence.topk_indices),
        ("logprobs", distillation.teacher_topk_logprobs, evidence.topk_logprobs),
        ("retained_mass", distillation.retained_mass, evidence.retained_mass),
        ("valid_mask", distillation.valid_mask, evidence.valid_mask),
    )
    for label, actual, expected in payload_pairs:
        if not torch.allclose(actual, expected, rtol=0, atol=0, equal_nan=True):
            raise ValueError(f"prepared distillation {label} does not match teacher evidence")
    if distillation.loss_weights.shape != response_mask.shape:
        raise ValueError("prepared distillation loss_weights must match trajectory response coordinates")
    _validate_loss_weights(distillation.loss_weights, "prepared distillation loss_weights")


def sampled_reverse_kl_loss(
    action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    distillation: SampledReverseKLInput,
    loss_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return an on-policy score-function surrogate for ``KL(policy || teacher)``."""
    expected_shape = action_log_probs.shape
    payload_tensors = (
        ("old_action_log_probs", old_action_log_probs),
        ("teacher_action_log_probs", distillation.teacher_action_log_probs),
        ("valid_mask", distillation.valid_mask),
        ("loss_weights", distillation.loss_weights),
    )
    for name, tensor in payload_tensors:
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must match action_log_probs shape {tuple(expected_shape)}")
    if distillation.valid_mask.dtype is not torch.bool:
        raise ValueError("distillation valid_mask must have bool dtype")
    expected_device = action_log_probs.device
    for name, tensor in payload_tensors:
        if tensor.device != expected_device:
            raise ValueError(f"{name} must be on the action_log_probs device {expected_device}")
    _validate_masked_logprobs(
        distillation.teacher_action_log_probs,
        distillation.valid_mask,
        "teacher action logprobs",
    )
    _validate_loss_weights(distillation.loss_weights, "distillation loss_weights")

    effective_mask = distillation.valid_mask
    if loss_mask is not None:
        if loss_mask.shape != expected_shape:
            raise ValueError(f"loss_mask must match action_log_probs shape {tuple(expected_shape)}")
        if loss_mask.device != expected_device:
            raise ValueError(f"loss_mask must be on the action_log_probs device {expected_device}")
        effective_mask = effective_mask & loss_mask.to(torch.bool)
    if not torch.any(effective_mask):
        raise ValueError("sampled reverse KL has no valid training tokens")

    teacher_logprobs = torch.where(
        effective_mask,
        distillation.teacher_action_log_probs,
        torch.zeros_like(action_log_probs),
    )
    behavior_logprobs = old_action_log_probs.detach()
    teacher_gap = behavior_logprobs - teacher_logprobs
    importance_ratio = safe_exp_delta(action_log_probs - behavior_logprobs, out_dtype=action_log_probs.dtype)
    token_loss = importance_ratio * teacher_gap * distillation.loss_weights
    return masked_mean(token_loss, effective_mask, dim=-1).mean()


def _validate_student_topk_surrogate_input(
    student_selected_logprobs: torch.Tensor,
    distillation: StudentTopKPolicySurrogateInput,
    loss_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Validate aligned selected-ID evidence and return the effective token mask."""
    indices = distillation.student_topk_indices
    if indices.ndim != 3 or indices.shape[-1] == 0 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("student-top-K indices must have integer [batch, response_len, K] shape with K > 0")
    expected_shape = indices.shape
    response_shape = expected_shape[:2]
    if distillation.valid_mask.shape != response_shape or distillation.valid_mask.dtype is not torch.bool:
        raise ValueError("student-top-K valid_mask must be boolean and match response coordinates")
    if distillation.loss_weights.shape != response_shape:
        raise ValueError("student-top-K loss_weights must match response coordinates")
    for name, tensor in (
        ("student_selected_logprobs", student_selected_logprobs),
        ("behavior_topk_logprobs", distillation.behavior_topk_logprobs),
        ("teacher_on_student_logprobs", distillation.teacher_on_student_logprobs),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must match student-top-K indices shape {expected_shape}")
        if tensor.device != indices.device:
            raise ValueError(f"{name} must be on the student-top-K indices device")
        _validate_masked_logprobs(tensor, distillation.valid_mask, name)
    if distillation.valid_mask.device != indices.device or distillation.loss_weights.device != indices.device:
        raise ValueError("student-top-K masks and weights must be on the selected-ID device")
    _validate_loss_weights(distillation.loss_weights, "student-top-K loss_weights")
    _validate_selected_token_ids(indices, distillation.valid_mask, "student-top-K")

    effective_mask = distillation.valid_mask
    if loss_mask is not None:
        if loss_mask.shape != response_shape or loss_mask.device != indices.device:
            raise ValueError("student-top-K loss_mask must match response coordinates and device")
        effective_mask = effective_mask & loss_mask.to(torch.bool)
    if not torch.any(effective_mask):
        raise ValueError("student-top-K OPD has no valid training tokens")
    return effective_mask


def student_topk_policy_surrogate_loss(
    student_selected_logprobs: torch.Tensor,
    distillation: StudentTopKPolicySurrogateInput,
    loss_mask: Optional[torch.Tensor],
    config: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the released Open-MOPD student-top-K clipped policy surrogate.

    This is not a KL divergence: the student-weighted rewards are detached, and
    the per-ID importance ratios receive dual clipping before summing over K.
    """
    if config.loss_reduction != TOKEN_MEAN_LOSS_REDUCTION:
        raise ValueError("student-top-K OPD requires token_mean loss reduction")
    clip_low = float(config.eps_clip_low)
    clip_high = float(config.eps_clip_high)
    clip_ratio_c = float(config.clip_ratio_c)
    if not all(math.isfinite(value) for value in (clip_low, clip_high, clip_ratio_c)):
        raise ValueError("student-top-K OPD clip ratios must be finite")
    if clip_low < 0 or clip_low >= 1 or clip_high < 0 or clip_ratio_c <= 1:
        raise ValueError("student-top-K OPD requires 0 <= clip_low < 1, clip_high >= 0, clip_ratio_c > 1")

    effective_mask = _validate_student_topk_surrogate_input(student_selected_logprobs, distillation, loss_mask)

    current = student_selected_logprobs[effective_mask]
    behavior = distillation.behavior_topk_logprobs[effective_mask]
    teacher = distillation.teacher_on_student_logprobs[effective_mask]
    student_weights = current.detach().float().softmax(dim=-1).to(current.dtype)
    advantages = -(current.detach() - teacher) * student_weights
    ratio = safe_exp_delta(current - behavior, out_dtype=current.dtype)
    raw_loss = -advantages * ratio
    clipped_loss = -advantages * ratio.clamp(1 - clip_low, 1 + clip_high)
    pessimistic_loss = torch.maximum(raw_loss, clipped_loss)
    dual_clipped_loss = torch.where(
        advantages < 0,
        torch.minimum(pessimistic_loss, -advantages * clip_ratio_c),
        pessimistic_loss,
    )
    per_token = dual_clipped_loss.sum(dim=-1) * distillation.loss_weights[effective_mask]
    loss = per_token.mean()
    metrics = {
        DISTILLATION_TOPK_METRIC: float(distillation.student_topk_indices.shape[-1]),
        "distillation_student_retained_mass_mean": behavior.exp().sum(dim=-1).mean().item(),
        "distillation_clip_fraction": (clipped_loss > raw_loss).float().mean().item(),
        "distillation_dual_clip_fraction": ((advantages < 0) & (pessimistic_loss > -advantages * clip_ratio_c))
        .float()
        .mean()
        .item(),
    }
    return loss, metrics


def student_topk_logprobs(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Return selected student logprobs, with NaN wherever the token ID is the invalid sentinel."""
    if logits.ndim != 3 or topk_indices.ndim != 3 or logits.shape[:2] != topk_indices.shape[:2]:
        raise ValueError("student logits and selected indices must have [batch, response_len, vocab/K] shapes")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected top-K indices must have integer dtype")
    if topk_indices.device != logits.device:
        raise ValueError("selected top-K indices must be on the student logits device")
    valid_indices = topk_indices != INVALID_TOPK_INDEX
    if torch.any(topk_indices < INVALID_TOPK_INDEX):
        raise ValueError(f"selected top-K indices cannot be smaller than the {INVALID_TOPK_INDEX} sentinel")
    if torch.any(topk_indices[valid_indices] >= logits.shape[-1]):
        raise ValueError(f"selected top-K indices must be smaller than student vocabulary size {logits.shape[-1]}")
    safe_indices = topk_indices.masked_fill(~valid_indices, 0).long()
    float_logits = logits.float()
    selected = torch.gather(float_logits, dim=-1, index=safe_indices)
    normalized = selected - torch.logsumexp(float_logits, dim=-1, keepdim=True)
    return normalized.masked_fill(~valid_indices, torch.nan)


def sparse_forward_kl_loss(
    student_topk_log_probs: torch.Tensor,
    distillation: SparseForwardKLInput,
    loss_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return conditional teacher top-K KL against the student's full distribution."""
    expected_shape = distillation.teacher_topk_indices.shape
    if student_topk_log_probs.shape != expected_shape:
        raise ValueError(f"student top-K logprobs must match teacher indices shape {tuple(expected_shape)}")
    if distillation.teacher_topk_logprobs.shape != expected_shape:
        raise ValueError("teacher top-K logprobs must match teacher indices")
    response_shape = expected_shape[:2]
    for name, tensor in (
        ("retained_mass", distillation.retained_mass),
        ("valid_mask", distillation.valid_mask),
        ("loss_weights", distillation.loss_weights),
    ):
        if tensor.shape != response_shape:
            raise ValueError(f"{name} must match response shape {tuple(response_shape)}")
    if distillation.valid_mask.dtype is not torch.bool:
        raise ValueError("distillation valid_mask must have bool dtype")
    expected_device = student_topk_log_probs.device
    for name, tensor in (
        ("teacher_topk_indices", distillation.teacher_topk_indices),
        ("teacher_topk_logprobs", distillation.teacher_topk_logprobs),
        ("retained_mass", distillation.retained_mass),
        ("valid_mask", distillation.valid_mask),
        ("loss_weights", distillation.loss_weights),
    ):
        if tensor.device != expected_device:
            raise ValueError(f"{name} must be on the student top-K logprobs device {expected_device}")
    _validate_topk_distribution(
        distillation.teacher_topk_indices,
        distillation.teacher_topk_logprobs,
        distillation.retained_mass,
        distillation.valid_mask,
    )
    _validate_masked_logprobs(student_topk_log_probs, distillation.valid_mask, "student top-K logprobs")
    _validate_loss_weights(distillation.loss_weights, "distillation loss_weights")
    effective_mask = distillation.valid_mask
    if loss_mask is not None:
        if loss_mask.shape != response_shape or loss_mask.device != expected_device:
            raise ValueError("loss_mask must match sparse forward KL response coordinates and device")
        effective_mask = effective_mask & loss_mask.to(torch.bool)
    if not torch.any(effective_mask):
        raise ValueError("sparse forward KL has no valid training tokens")

    teacher_logprobs = distillation.teacher_topk_logprobs[effective_mask].float()
    student_logprobs = student_topk_log_probs[effective_mask].float()
    retained_mass = distillation.retained_mass[effective_mask].float()
    conditional_teacher_logprobs = teacher_logprobs - retained_mass.log().unsqueeze(-1)
    conditional_teacher_probs = conditional_teacher_logprobs.exp()
    per_token_kl = torch.sum(
        conditional_teacher_probs * (conditional_teacher_logprobs - student_logprobs),
        dim=-1,
    )
    weights = distillation.loss_weights[effective_mask].float()
    token_loss = torch.zeros(response_shape, dtype=per_token_kl.dtype, device=expected_device)
    token_loss[effective_mask] = per_token_kl * weights
    loss = masked_mean(token_loss, effective_mask, dim=-1).mean()
    metrics = {
        "distillation_retained_mass_mean": retained_mass.mean().item(),
        "distillation_retained_mass_min": retained_mass.min().item(),
        DISTILLATION_TOPK_METRIC: float(expected_shape[-1]),
    }
    return loss, metrics
