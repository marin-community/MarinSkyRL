"""Detached correction masks for advantage-linear policy objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from skyrl_train.utils.policy_math import masked_mean


METRIC_KEYS = (
    "offpolicy_mask/masked_fraction",
    "offpolicy_mask/masked_fraction_low",
    "offpolicy_mask/masked_fraction_high",
    "offpolicy_mask/vetoed_sequence_fraction",
)


@dataclass(frozen=True)
class MaskResult:
    advantages: torch.Tensor
    loss_mask: torch.Tensor | None
    metrics: dict[str, float]


def validate_offpolicy_masks(config: DictConfig, *, global_loss_denom: float | None = None) -> None:
    """Fail closed for unsupported objective, ratio, and normalization combinations."""

    off = config.get("offpolicy_mask", {})
    if not off.get("enabled", False):
        return
    if config.policy_loss_type not in ("regular", "dual_clip", "behavior_clip", "cispo", "gspo"):
        raise ValueError("offpolicy_mask requires an advantage-linear policy loss")
    if config.get("use_tis", False) and config.policy_loss_type not in ("regular", "dual_clip"):
        raise ValueError("TIS composes only with regular or dual_clip")
    if off.get("renormalize", False) and (
        global_loss_denom is not None or config.loss_reduction == "seq_mean_token_sum_norm_global"
    ):
        raise ValueError("offpolicy_mask renormalization does not support a fixed global denominator")
    if off.ratio not in ("mismatch", "full"):
        raise ValueError("offpolicy_mask ratio must be mismatch or full")
    if not all(math.isfinite(float(off[key])) for key in ("low", "high", "veto_ratio")):
        raise ValueError("offpolicy_mask bounds must be finite")
    if not (0 < off.veto_ratio <= off.low <= off.high):
        raise ValueError("offpolicy_mask requires 0 < veto_ratio <= low <= high")


def _masked_fraction(condition: torch.Tensor, selected: torch.Tensor) -> float:
    return float(masked_mean(condition.float(), selected).mean().detach().item())


def apply_offpolicy_masks(
    *,
    action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    rollout_logprobs: torch.Tensor | None,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None,
    config: DictConfig,
) -> MaskResult:
    """Mask detached mismatch outliers while preserving the configured denominator."""

    off = config.get("offpolicy_mask", {})
    metrics = dict.fromkeys(METRIC_KEYS, 0.0)
    if not off.get("enabled", False):
        return MaskResult(advantages, loss_mask, metrics)
    if rollout_logprobs is None:
        raise ValueError("offpolicy_mask requires rollout logprobs")
    selected = torch.ones_like(advantages, dtype=torch.bool) if loss_mask is None else loss_mask > 0
    numerator = old_action_log_probs if off.ratio == "mismatch" else action_log_probs
    delta = (numerator - rollout_logprobs).detach()
    if not torch.isfinite(delta[selected]).all():
        raise ValueError("offpolicy_mask selected log ratios must be finite")
    ratio = delta.clamp(-20, 20).exp()
    low = selected & (ratio < off.low)
    high = selected & (ratio > off.high)
    veto = (selected & (ratio < off.veto_ratio)).any(dim=-1, keepdim=True)
    removed = selected & (low | high | veto)
    metrics.update(
        {
            "offpolicy_mask/masked_fraction": _masked_fraction(removed, selected),
            "offpolicy_mask/masked_fraction_low": _masked_fraction(low, selected),
            "offpolicy_mask/masked_fraction_high": _masked_fraction(high, selected),
            "offpolicy_mask/vetoed_sequence_fraction": _masked_fraction(veto.squeeze(-1), selected.any(-1)),
        }
    )
    if off.renormalize:
        weights = torch.ones_like(advantages) if loss_mask is None else loss_mask
        return MaskResult(advantages, torch.where(removed, torch.zeros_like(weights), weights), metrics)
    return MaskResult(torch.where(removed, torch.zeros_like(advantages), advantages), loss_mask, metrics)
