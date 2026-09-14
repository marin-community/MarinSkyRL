"""Teacher evidence contracts and backend-neutral distillation objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Self

import torch

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.tensor_math import masked_mean, safe_exp_delta


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


TeacherEvidenceBatch = ChosenTokenTeacherEvidence | TopKTeacherEvidence


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


def sampled_reverse_kl_input_from_tensors(
    teacher_action_log_probs: Optional[torch.Tensor],
    valid_mask: Optional[torch.Tensor],
    loss_weights: Optional[torch.Tensor],
) -> Optional[SampledReverseKLInput]:
    """Build an active learner payload, rejecting partially propagated evidence."""
    values = (teacher_action_log_probs, valid_mask, loss_weights)
    if all(value is None for value in values):
        return None
    if teacher_action_log_probs is None or valid_mask is None or loss_weights is None:
        raise ValueError("sampled reverse KL requires teacher logprobs, valid_mask, and loss_weights together")
    return SampledReverseKLInput(
        teacher_action_log_probs=teacher_action_log_probs,
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


def validate_teacher_evidence(request: TeacherScoreRequest, evidence: TeacherEvidenceBatch) -> None:
    """Reject evidence whose identity, token coordinates, shape, or values are ambiguous."""
    validate_teacher_score_request(request)
    if not isinstance(evidence, (ChosenTokenTeacherEvidence, TopKTeacherEvidence)):
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

    if evidence.topk_indices.shape != evidence.topk_logprobs.shape or evidence.topk_indices.ndim != 3:
        raise ValueError("teacher top-K indices and logprobs must have matching [batch, response_len, K] shapes")
    if evidence.topk_indices.shape[2] == 0:
        raise ValueError("teacher top-K evidence must retain at least one token per valid position")
    if evidence.topk_indices.shape[:2] != evidence.valid_mask.shape:
        raise ValueError("teacher top-K evidence must match valid_mask response coordinates")
    if evidence.retained_mass.shape != evidence.valid_mask.shape:
        raise ValueError("teacher retained_mass must match valid_mask")
    if evidence.topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("teacher top-K indices must have integer dtype")
    _validate_masked_logprobs(evidence.topk_logprobs, evidence.valid_mask, "teacher top-K logprobs")
    if not torch.is_floating_point(evidence.retained_mass):
        raise ValueError("teacher retained_mass must have floating-point dtype")
    valid_indices = evidence.topk_indices[evidence.valid_mask]
    valid_mass = evidence.retained_mass[evidence.valid_mask]
    if torch.any(valid_indices < 0):
        raise ValueError("valid teacher top-K indices must be non-negative")
    if not torch.all(torch.isfinite(valid_mass)) or torch.any((valid_mass <= 0) | (valid_mass > 1)):
        raise ValueError("valid teacher retained_mass must lie in (0, 1]")
    if not torch.all(torch.isnan(evidence.retained_mass[~evidence.valid_mask])):
        raise ValueError("invalid teacher retained_mass must be NaN")
    if not torch.all(evidence.topk_indices[~evidence.valid_mask] == -1):
        raise ValueError("invalid teacher top-K indices must use the -1 sentinel")


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
