"""Detached, microbatch-scoped correction transforms for advantage-linear losses."""

import math
from dataclasses import dataclass

import torch
from omegaconf import DictConfig


METRIC_KEYS = (
    "offpolicy_mask/masked_fraction",
    "offpolicy_mask/masked_fraction_low",
    "offpolicy_mask/masked_fraction_high",
    "offpolicy_mask/vetoed_sequence_fraction",
    "offpolicy_mask/masked_advantage_abs_mean",
    "offpolicy_mask/masked_entropy_mean",
    "m2_mask/m2_before",
    "m2_mask/m2_after",
    "m2_mask/masked_fraction",
    "m2_mask/candidate_fraction",
    "m2_mask/unsatisfied",
    "m2_mask/tau_effective",
    "m2_mask/masked_entropy_mean",
    "entropy_mean_selected",
)


@dataclass(frozen=True)
class MaskResult:
    advantages: torch.Tensor
    loss_mask: torch.Tensor | None
    metrics: dict[str, float]
    clip_bounds: tuple[torch.Tensor | float, torch.Tensor | float] | None = None


def validate_offpolicy_masks(config: DictConfig, *, global_loss_denom: float | None = None) -> None:
    """Fail closed for unsupported objective/ratio/normalization combinations."""
    off = config.get("offpolicy_mask", {})
    m2 = config.get("m2_mask", {})
    enabled = [item for item in (off, m2) if item.get("enabled", False)]
    if not enabled:
        return
    if config.policy_loss_type not in ("regular", "dual_clip", "behavior_clip", "cispo", "gspo"):
        raise ValueError("correction masks require an advantage-linear policy loss")
    if config.get("use_tis", False) and config.policy_loss_type not in ("regular", "dual_clip"):
        raise ValueError("TIS composes only with regular or dual_clip")
    if any(item.get("renormalize", False) for item in enabled) and (
        global_loss_denom is not None or config.loss_reduction == "seq_mean_token_sum_norm_global"
    ):
        raise ValueError("renormalize=true does not support an externally fixed global denominator")
    if off.get("enabled", False):
        if off.ratio not in ("mismatch", "full"):
            raise ValueError("offpolicy_mask ratio must be mismatch or full")
        if not all(math.isfinite(float(off[key])) for key in ("low", "high", "veto_ratio")):
            raise ValueError("offpolicy_mask bounds must be finite")
        if not (0 < off.veto_ratio <= off.low <= off.high):
            raise ValueError("offpolicy_mask requires 0 < veto_ratio <= low <= high")
    if m2.get("enabled", False):
        if m2.ratio not in ("stale", "full") or m2.mode not in ("mask", "clip"):
            raise ValueError("m2_mask requires ratio stale/full and mode mask/clip")
        if not math.isfinite(float(m2.tau)) or m2.tau <= 0:
            raise ValueError("m2_mask tau must be finite and positive")
        if m2.mode == "clip" and config.policy_loss_type not in ("regular", "dual_clip"):
            raise ValueError("m2_mask clip mode requires regular or dual_clip")


def _mean_selected(values: torch.Tensor, selected: torch.Tensor) -> float:
    return float(values.detach()[selected].double().mean().item()) if selected.any() else 0.0


def _remove(advantages, loss_mask, removed, *, renormalize):
    if renormalize:
        weights = torch.ones_like(advantages) if loss_mask is None else loss_mask
        return advantages, torch.where(removed, torch.zeros_like(weights), weights)
    return torch.where(removed, torch.zeros_like(advantages), advantages), loss_mask


def minimal_m2_mask(delta: torch.Tensor, advantages: torch.Tensor, selected: torch.Tensor, tau: float):
    """Remove the smallest descending harmful prefix with retained mean strictly below tau."""
    delta = delta.detach()
    squared = delta.double().square()
    candidates = selected & (((advantages > 0) & (delta > 0)) | ((advantages < 0) & (delta < 0)))
    count = int(selected.sum())
    removed = torch.zeros_like(selected)
    if count == 0:
        return removed, candidates, 0.0, 0.0, False
    total = squared[selected].sum()
    before = float(total / count)
    if before < tau:
        return removed, candidates, before, before, False
    indices = candidates.flatten().nonzero().flatten()
    values = squared.flatten()[indices]
    order = torch.argsort(values, descending=True, stable=True)
    prefix = torch.cat([values.new_zeros(1), values[order].cumsum(0)])
    remaining = count - torch.arange(prefix.numel(), device=delta.device)
    means = (total - prefix).clamp_min(0) / remaining.clamp_min(1)
    valid = (remaining > 0) & (means < tau)
    # The retained mean can turn upwards; binary search is not justified.
    positions = valid.nonzero().flatten()
    satisfied = bool(positions.numel())
    take = int(positions[0]) if satisfied else indices.numel()
    removed.flatten()[indices[order[:take]]] = True
    after = _mean_selected(squared, selected & ~removed)
    return removed, candidates, before, after, not satisfied


def released_m2_clip_bounds(delta: torch.Tensor, advantages: torch.Tensor, selected: torch.Tensor, budget: float):
    """Reproduce M2PO af54a3e's discrete clip bounds, including its float32 exp.

    The reference selects the preceding sorted breakpoint, not a continuous
    water-filling solution. Its harmful-quadrant sign uses advantages[:, 0].
    Report the actual clipped harmful moment rather than its k=0 logging bug.
    """
    detached = delta.detach()
    ratio = detached.exp()
    sign = advantages.detach()[:, :1]
    harmful = selected & (((sign > 1e-12) & (ratio > 1 + 1e-12)) | ((sign < -1e-12) & (ratio < 1 - 1e-12)))
    values = detached.square()[harmful]
    if values.numel() == 0:
        return (0.0, 100000.0), harmful, 0.0, 0.0, 100000.0
    before = float(values.sum().item() / values.numel())
    if before <= budget + 1e-12:
        return (0.0, 100000.0), harmful, before, before, 100000.0
    ordered = values.sort().values
    cumulative = ordered.cumsum(0)
    rest = ordered.numel() - torch.arange(ordered.numel(), device=ordered.device) - 1
    target = budget * ordered.numel()
    if target <= 1e-12:
        raise ValueError("released M2 clip reference has no tuple result at this tiny target budget")
    valid = (ordered.double() - 1e-12) * rest + cumulative.double() >= target - 1e-12
    positions = valid.nonzero().flatten()
    if positions.numel() == 0:
        raise ValueError("released M2 threshold has no finite breakpoint")
    index = int(positions[0])
    threshold_square = 0.0 if index == 0 else float(ordered[index - 1]) - 1e-12
    if threshold_square < 0:
        raise ValueError("released M2 clip reference has a non-real threshold for this microbatch")
    threshold = math.sqrt(threshold_square)
    bounds = (float(torch.tensor(-threshold).exp()), float(torch.tensor(threshold).exp()))
    after = float(values.double().clamp_max(threshold * threshold).mean())
    return bounds, harmful, before, after, threshold


def apply_offpolicy_masks(
    *,
    action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    rollout_logprobs: torch.Tensor | None,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor | None,
    token_entropy: torch.Tensor,
    config: DictConfig,
) -> MaskResult:
    """Apply masks before policy loss; disabled mode preserves input identities."""
    off = config.get("offpolicy_mask", {})
    m2 = config.get("m2_mask", {})
    metrics = dict.fromkeys(METRIC_KEYS, 0.0)
    if not off.get("enabled", False) and not m2.get("enabled", False):
        return MaskResult(advantages, loss_mask, metrics)
    selected = torch.ones_like(advantages, dtype=torch.bool) if loss_mask is None else loss_mask > 0
    metrics["entropy_mean_selected"] = _mean_selected(token_entropy, selected)
    original_advantages = advantages
    bounds = None
    if off.get("enabled", False):
        if rollout_logprobs is None:
            raise ValueError("offpolicy_mask requires rollout logprobs")
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
                "offpolicy_mask/masked_fraction": _mean_selected(removed.float(), selected),
                "offpolicy_mask/masked_fraction_low": _mean_selected(low.float(), selected),
                "offpolicy_mask/masked_fraction_high": _mean_selected(high.float(), selected),
                "offpolicy_mask/vetoed_sequence_fraction": _mean_selected(veto.squeeze(-1).float(), selected.any(-1)),
                "offpolicy_mask/masked_advantage_abs_mean": _mean_selected(original_advantages.abs(), removed),
                "offpolicy_mask/masked_entropy_mean": _mean_selected(token_entropy, removed),
            }
        )
        advantages, loss_mask = _remove(advantages, loss_mask, removed, renormalize=off.renormalize)
    if m2.get("enabled", False):
        denominator = old_action_log_probs if m2.ratio == "stale" else rollout_logprobs
        if denominator is None:
            raise ValueError("full-ratio m2_mask requires rollout logprobs")
        delta = (action_log_probs - denominator).detach()
        selected = torch.ones_like(advantages, dtype=torch.bool) if loss_mask is None else loss_mask > 0
        if not torch.isfinite(delta[selected]).all():
            raise ValueError("m2_mask selected log ratios must be finite")
        if m2.mode == "mask":
            removed, candidates, before, after, unsatisfied = minimal_m2_mask(delta, advantages, selected, m2.tau)
            advantages, loss_mask = _remove(advantages, loss_mask, removed, renormalize=m2.renormalize)
            metrics["m2_mask/masked_fraction"] = _mean_selected(removed.float(), selected)
            metrics["m2_mask/masked_entropy_mean"] = _mean_selected(token_entropy, removed)
            metrics["m2_mask/unsatisfied"] = float(unsatisfied)
        else:
            bounds, candidates, before, after, threshold = released_m2_clip_bounds(delta, advantages, selected, m2.tau)
            metrics["m2_mask/tau_effective"] = threshold
            if m2.ratio == "full":
                conversion = (rollout_logprobs - old_action_log_probs).detach().exp()
                bounds = (bounds[0] * conversion, bounds[1] * conversion)
        metrics.update(
            {
                "m2_mask/m2_before": before,
                "m2_mask/m2_after": after,
                "m2_mask/candidate_fraction": _mean_selected(candidates.float(), selected),
            }
        )
    return MaskResult(advantages, loss_mask, metrics, bounds)
