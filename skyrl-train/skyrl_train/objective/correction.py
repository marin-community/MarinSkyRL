from dataclasses import dataclass
from collections.abc import Callable
import math

import torch

from skyrl_train.config.objective_spec import (
    CorrectionAction,
    OffPolicyCorrection,
    SequenceAggregate,
    SequenceRule,
)
from skyrl_train.metric_names import (
    CORRECTION_WEIGHT_MEAN_METRIC,
    CORRECTION_TRUNCATED_FRACTION_METRIC,
    CORRECTION_MASKED_FRACTION_METRIC,
)
from skyrl_train.tensor_math import LOG_PROB_DELTA_CLIP


@dataclass(frozen=True)
class CorrectionResult:
    weights: torch.Tensor
    metrics: dict[str, float]


@torch.no_grad()
def compute_correction(
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    correction: OffPolicyCorrection,
    *,
    all_reduce: Callable | None = None,
    contributes: bool = True,
) -> CorrectionResult:
    """Return detached policy numerator weights and token-weighted correction statistics."""
    assert old_log_probs.shape == rollout_log_probs.shape == loss_mask.shape
    valid = loss_mask > 0
    old = torch.where(valid, old_log_probs, 0).float()
    rollout = torch.where(valid, rollout_log_probs, 0).float()
    log_ratio = old - rollout
    weights = torch.ones_like(log_ratio)
    masked = torch.zeros_like(valid)
    truncated = torch.zeros_like(valid)
    for rule in correction.rules:
        delta = log_ratio
        if isinstance(rule, SequenceRule) and rule.aggregate is not SequenceAggregate.EXTREME_TOKEN:
            delta = log_ratio.sum(dim=-1, keepdim=True)
            if rule.aggregate is SequenceAggregate.GEOMETRIC:
                delta = delta / valid.sum(dim=-1, keepdim=True).clamp(min=1)
            else:
                delta = delta.clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
        if rule.action is CorrectionAction.TRUNCATE:
            log_high = math.log(rule.high)
            weights *= delta.clamp(max=log_high).exp()
            truncated |= delta > log_high
        else:
            keep = torch.ones_like(delta, dtype=torch.bool)
            if rule.low is not None:
                keep &= delta >= math.log(rule.low)
            if rule.high is not None:
                keep &= delta <= math.log(rule.high)
            if isinstance(rule, SequenceRule) and rule.aggregate is SequenceAggregate.EXTREME_TOKEN:
                keep = (keep | ~valid).all(dim=-1, keepdim=True)
            weights *= keep
            masked |= ~keep
    weights = torch.where(valid, weights, 0)
    totals = torch.stack(
        [
            valid.sum(),
            (weights.double() if all_reduce is not None else weights).sum(),
            (truncated & valid).sum(),
            (masked & valid).sum(),
        ]
    )
    if not contributes:
        totals.zero_()
    if all_reduce is not None:
        totals = all_reduce(totals, "sum")
    count = totals[0].clamp(min=1)
    metrics = (
        {
            CORRECTION_WEIGHT_MEAN_METRIC: (totals[1] / count).item(),
            CORRECTION_TRUNCATED_FRACTION_METRIC: (totals[2] / count).item(),
            CORRECTION_MASKED_FRACTION_METRIC: (totals[3] / count).item(),
        }
        if correction.rules
        else {}
    )
    return CorrectionResult(weights, metrics)
