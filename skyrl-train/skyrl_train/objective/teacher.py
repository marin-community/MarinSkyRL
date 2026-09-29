from dataclasses import replace

import torch

from marinskyrl.distillation import DistillationObjectiveKind
from skyrl_train.config.objective_spec import TopKLossParams
from skyrl_train.distillation import TeacherTopKInput, TopKEvidence
from skyrl_train.objective.losses import TokenLoss
from skyrl_train.tensor_math import safe_exp_delta


@torch.no_grad()
def teacher_advantages(
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    route_weights: torch.Tensor,
    clip: float | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return detached teacher credit; route weights include the teacher coefficient."""
    assert teacher_log_probs.shape == old_log_probs.shape == valid_mask.shape == route_weights.shape
    valid = valid_mask.bool()
    teacher = torch.where(valid, teacher_log_probs, 0)
    old = torch.where(valid, old_log_probs, 0)
    weights = torch.where(valid, route_weights, 0)
    delta = teacher - old
    clipped = delta if clip is None else delta.clamp(-clip, clip)
    advantage = clipped * weights
    count = valid.sum()
    denominator = count.clamp(min=1)
    return advantage, {
        "distillation/teacher_advantage_mean": (advantage.sum() / denominator).item(),
        "distillation/teacher_advantage_abs_mean": (advantage.abs().sum() / denominator).item(),
        "distillation/teacher_advantage_clipped_fraction": ((clipped != delta).sum() / denominator).item(),
        "distillation/valid_tokens": count.item(),
    }


def mask_teacher_evidence(evidence: TopKEvidence, loss_mask: torch.Tensor) -> TopKEvidence:
    """Use response eligibility as well as teacher validity for the objective row."""
    assert evidence.valid_mask.shape == loss_mask.shape
    return replace(evidence, valid_mask=evidence.valid_mask & (loss_mask > 0))


def topk_teacher_loss(
    evidence: TopKEvidence, student_log_probs_on_support: torch.Tensor, params: TopKLossParams
) -> TokenLoss:
    """Return teacher values per response token, before route weighting and reduction."""
    valid = evidence.valid_mask
    assert student_log_probs_on_support.shape[:-1] == valid.shape
    # Invalid transport positions contain NaN. Sanitize before nonlinear operations
    # so those positions have finite, zero gradients as well as zero loss values.
    selected = valid.unsqueeze(-1)
    current = torch.where(selected, student_log_probs_on_support, 0)
    denominator = valid.sum().clamp(min=1)
    metrics = {"distillation_topk": float(current.shape[-1])}
    if isinstance(evidence, TeacherTopKInput):
        if params.objective is not DistillationObjectiveKind.SPARSE_FORWARD_KL:
            raise ValueError(f"teacher-support evidence cannot use {params.objective}")
        assert evidence.teacher_topk_logprobs.shape == current.shape
        teacher = torch.where(selected, evidence.teacher_topk_logprobs, 0).float()
        mass = torch.where(valid, evidence.retained_mass, 1).float()
        conditional = teacher - mass.log().unsqueeze(-1)
        values = (conditional.exp() * (conditional - current.float())).sum(-1)
        metrics.update(
            distillation_retained_mass_mean=(torch.where(valid, mass, 0).sum() / denominator).item(),
            distillation_retained_mass_min=mass[valid].min().item() if valid.any() else 0.0,
        )
    else:
        if params.objective is not DistillationObjectiveKind.STUDENT_TOPK_POLICY_SURROGATE:
            raise ValueError(f"student-support evidence cannot use {params.objective}")
        assert evidence.behavior_topk_logprobs.shape == evidence.teacher_on_student_logprobs.shape == current.shape
        behavior = torch.where(selected, evidence.behavior_topk_logprobs, 0)
        teacher = torch.where(selected, evidence.teacher_on_student_logprobs, 0)
        student_weights = current.detach().float().softmax(dim=-1).to(current.dtype)
        advantages = -(current.detach() - teacher) * student_weights
        ratio = safe_exp_delta(current - behavior, out_dtype=current.dtype)
        raw = -advantages * ratio
        clipped = -advantages * ratio.clamp(1 - params.eps_clip_low, 1 + params.eps_clip_high)
        pessimistic = torch.maximum(raw, clipped)
        bound = -advantages * params.clip_ratio_c
        values = torch.where(advantages < 0, torch.minimum(pessimistic, bound), pessimistic).sum(-1)
        entry_count = denominator * current.shape[-1]
        metrics.update(
            distillation_student_retained_mass_mean=(
                torch.where(valid, behavior.exp().sum(-1), 0).sum() / denominator
            ).item(),
            distillation_clip_fraction=((clipped > raw) & selected).sum().div(entry_count).item(),
            distillation_dual_clip_fraction=(
                ((advantages < 0) & (pessimistic > bound) & selected).sum().div(entry_count).item()
            ),
        )
    return TokenLoss(torch.where(valid, values, 0), metrics)
