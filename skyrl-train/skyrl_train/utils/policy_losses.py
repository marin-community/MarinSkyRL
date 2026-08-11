"""Policy loss implementations and clipping diagnostics.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional

import loguru
import torch
from omegaconf import DictConfig

from skyrl_train.utils.loss_reduction import reduce_loss
from skyrl_train.utils.algorithm_registry import PolicyLossType, register_policy_loss
from skyrl_train.utils.policy_math import LOG_PROB_DELTA_CLIP, masked_mean, safe_exp_delta


@dataclass(frozen=True)
class PolicyClipMetrics:
    """Stable clipping diagnostic record emitted by every training backend."""

    ppo_clip_ratio: float = 0.0
    ppo_clip_ratio_low: float = 0.0
    ppo_clip_ratio_high: float = 0.0
    ppo_clip_pressure_low: float = 0.0
    ppo_clip_pressure_high: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


POLICY_CLIP_METRIC_KEYS = tuple(field.name for field in fields(PolicyClipMetrics))


def _masked_fraction(condition: torch.Tensor, loss_mask: Optional[torch.Tensor]) -> float:
    return masked_mean(condition.float(), loss_mask).mean().detach().item()


def clipping_metrics(
    ratio: torch.Tensor,
    selected: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    *,
    eps_clip_low: float,
    eps_clip_high: float,
    pooled_clip_ratio: Optional[float] = None,
) -> dict[str, float]:
    """Partition clipping decisions and pre-clamp pressure by ratio side.

    The low/high clip ratios count selected tokens beyond the matching bound;
    pressure counts every token beyond each bound before objective selection.
    The pooled ratio counts all selected tokens unless the caller supplies its
    own loss-specific ``pooled_clip_ratio``.
    """
    low_pressure = ratio < 1 - eps_clip_low
    high_pressure = ratio > 1 + eps_clip_high
    low_selected = selected & low_pressure
    high_selected = selected & high_pressure
    if pooled_clip_ratio is None:
        pooled_clip_ratio = _masked_fraction(selected, loss_mask)
    return PolicyClipMetrics(
        ppo_clip_ratio=pooled_clip_ratio,
        ppo_clip_ratio_low=_masked_fraction(low_selected, loss_mask),
        ppo_clip_ratio_high=_masked_fraction(high_selected, loss_mask),
        ppo_clip_pressure_low=_masked_fraction(low_pressure, loss_mask),
        ppo_clip_pressure_high=_masked_fraction(high_pressure, loss_mask),
    ).as_dict()


def complete_clip_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Fill omitted clipping diagnostics at the training-backend boundary."""
    return PolicyClipMetrics().as_dict() | metrics


@register_policy_loss(PolicyLossType.REGULAR)
@register_policy_loss(PolicyLossType.DUAL_CLIP)
def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    assert config.policy_loss_type in ["regular", "dual_clip"], "loss_type must be either 'regular' or 'dual_clip'"
    loss_reduction = config.loss_reduction
    assert loss_reduction in [
        "token_mean",
        "sequence_mean",
        "seq_mean_token_sum_norm",
        "seq_mean_token_sum_norm_global",
    ], (
        "loss_reduction must be 'token_mean', 'sequence_mean', 'seq_mean_token_sum_norm', "
        "or 'seq_mean_token_sum_norm_global'"
    )

    ratio = safe_exp_delta(log_probs - old_log_probs, out_dtype=log_probs.dtype)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_metrics = clipping_metrics(
        ratio,
        -surr2 > -surr1,
        loss_mask,
        eps_clip_low=config.eps_clip_low,
        eps_clip_high=config.eps_clip_high,
    )
    clip_pg_losses1 = loss
    if config.policy_loss_type == "dual_clip":
        pg_losses3 = -advantages * config.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Missing rollout logprobs degrade this batch to the standard policy loss;
    # the trainer reports `tis/batch_skipped_no_logprobs` separately.
    if config.use_tis and rollout_logprobs is not None:
        loguru.logger.debug(f"Using TIS with dtype: {rollout_logprobs.dtype}")
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = safe_exp_delta(old_log_probs - rollout_logprobs, out_dtype=log_probs.dtype)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        loss = loss * tis_imp_ratio

    loss = reduce_loss(
        loss,
        loss_mask,
        loss_reduction,
        config.max_seq_len,
        global_denom=getattr(config, "global_loss_denom", None),
    )
    return loss, clip_metrics


@register_policy_loss(PolicyLossType.SAPO)
def sapo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    SAPO (Soft Adaptive Policy Optimization) policy loss function.

    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    """
    # SAPO has only been established as stable with sequence_mean reduction.
    loss_reduction = config.loss_reduction
    if loss_reduction != "sequence_mean":
        loguru.logger.warning(f"With SAPO it's recommended to use 'sequence_mean' loss reduction; got {loss_reduction}")

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.sapo.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.sapo.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # Compute token-level importance sampling in log space.
    log_ratio = log_probs - old_log_probs

    # Clamp log_ratio for stability -> avoid overflow in exp()
    log_ratio = torch.clamp(log_ratio, min=-LOG_PROB_DELTA_CLIP, max=LOG_PROB_DELTA_CLIP)

    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(log_ratio)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    loss = -gates * advantages

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len)

    return loss, {}


@register_policy_loss(PolicyLossType.GSPO)
def gspo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    GSPO (Group Sequence Policy Optimization) policy loss function,
    as proposed in https://arxiv.org/abs/2507.18071.

    This implements sequence-level importance sampling instead of token-level importance sampling.
    The key difference is that importance weights are computed at the sequence level and then
    applied uniformly across all tokens in the sequence. This can lead to more stable training
    dynamics by reducing the variance in clipping behavior within sequences.

    The variant of GSPO used here is GSPO-token, a generalization which allows for token-level
    advantages [equations 14 and 15 in the paper].
    """
    # GSPO has only been established as stable with sequence_mean reduction.
    loss_reduction = config.loss_reduction
    if loss_reduction != "sequence_mean":
        loguru.logger.warning(f"With GSPO it's recommended to use 'sequence_mean' loss reduction; got {loss_reduction}")

    # Compute log ratios
    log_ratio = log_probs - old_log_probs

    # Key GSPO innovation: sequence-level importance sampling
    # Instead of using per-token ratios, compute sequence-averaged ratios
    log_importance_weights = masked_mean(log_ratio, loss_mask, dim=-1).unsqueeze(-1)

    # Preserve the token gradient while applying the detached sequence-level weight.
    # The operation order avoids precision loss (volcengine/verl#2775).
    log_token_importance_weights = log_probs - log_probs.detach() + log_importance_weights.detach()
    # clip to avoid overflow
    log_token_importance_weights = torch.clamp(log_token_importance_weights, max=10)
    ratio = torch.exp(log_token_importance_weights)

    # Standard PPO surrogate objective with sequence-level importance weights
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)

    clip_metrics = clipping_metrics(
        ratio,
        -surr2 > -surr1,
        loss_mask,
        eps_clip_low=config.eps_clip_low,
        eps_clip_high=config.eps_clip_high,
    )

    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len)

    return loss, clip_metrics


@register_policy_loss(PolicyLossType.CISPO)
def compute_policy_loss_cispo(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Implementation of CISPO (Clipped IS-weight Policy Optimization) loss function,
    as proposed in https://arxiv.org/abs/2506.13585.

    Instead of clipping the importance sampling ratio in the loss directly, as done
    in PPO loss, CISPO clips the importance sampling ratio in the policy gradient
    update. This means the model can still learn from samples whose importance sampling
    ratio is clipped in CISPO, as opposed to PPO where these samples have zero
    gradient and are essentially ignored.
    """
    ratio = safe_exp_delta(log_probs - old_log_probs, out_dtype=log_probs.dtype)
    clamped_ratio = torch.clamp(ratio, 1 - config.cispo.cispo_eps_clip_low, 1 + config.cispo.cispo_eps_clip_high)
    loss = -advantages * clamped_ratio.detach() * log_probs

    is_clipped = (ratio < 1 - config.cispo.cispo_eps_clip_low) | (ratio > 1 + config.cispo.cispo_eps_clip_high)
    clip_metrics = clipping_metrics(
        ratio,
        is_clipped,
        loss_mask,
        eps_clip_low=config.cispo.cispo_eps_clip_low,
        eps_clip_high=config.cispo.cispo_eps_clip_high,
    )

    loss = reduce_loss(loss, loss_mask, config.loss_reduction, config.max_seq_len)
    return loss, clip_metrics


@register_policy_loss(PolicyLossType.CLIP_COV)
def compute_policy_loss_clip_cov(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clip-Cov policy loss function implementation.

    Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    This method combines standard PPO clipping with covariance-based clipping
    to provide more stable training dynamics.
    """
    # Extract config parameters with defaults
    clip_cov_ratio = config.clip_cov.clip_ratio
    clip_cov_lb = config.clip_cov.clip_cov_lb
    clip_cov_ub = config.clip_cov.clip_cov_ub

    negative_approx_kl = log_probs - old_log_probs
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio

    pg_losses2 = -advantages * torch.clamp(ratio, 1 - config.eps_clip_low, 1 + config.eps_clip_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (loss_mask > 0)

    # Compute covariance for clipping decision
    cov_all = (advantages - masked_mean(advantages, loss_mask)) * (
        log_probs - masked_mean(log_probs.detach(), loss_mask)
    )
    cov_all[loss_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    # Determine number of tokens to clip based on clip_ratio
    clip_num = max(int(clip_cov_ratio * loss_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (loss_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    # Create correction mask
    corr = torch.ones_like(advantages)
    if len(top_k_idx) > 0:
        corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    # Compute clip fraction for monitoring
    clip_frac = masked_mean((corr == 0).float(), loss_mask)

    # Apply correction mask to losses
    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = reduce_loss(
        loss=pg_losses,
        loss_mask=loss_mask,
        loss_reduction=config.loss_reduction,
        max_seq_len=config.max_seq_len,
    )

    return pg_loss, clipping_metrics(
        ratio,
        clip_by_origin,
        loss_mask,
        eps_clip_low=config.eps_clip_low,
        eps_clip_high=config.eps_clip_high,
        pooled_clip_ratio=clip_frac.item(),
    )


@register_policy_loss(PolicyLossType.KL_COV)
def compute_policy_loss_kl_cov(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """KL-Cov policy loss function implementation.

    Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Uses covariance-based selection to apply KL regularization to a subset of tokens.
    """
    kl_cov_frac = config.kl_cov.kl_cov_frac  # This should be a percentage (e.g., 0.2 for 20%)
    ppo_kl_coef = config.kl_cov.ppo_kl_coef

    negative_approx_kl = log_probs - old_log_probs
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1.clone()

    all_valid = loss_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_probs[all_valid].detach().reshape(-1).cpu()

    if len(all_valid_adv) > 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        # Use percentage-based selection like the reference implementation
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_frac))

        if k_percent_nums > 0:
            large_cov_idxs = torch.topk(cov_lst_all, min(k_percent_nums, len(cov_lst_all)), largest=True).indices

            if len(large_cov_idxs) > 0:
                large_cov_idxs = all_valid_idx[large_cov_idxs]
                pg_losses[
                    large_cov_idxs // advantages.shape[1],
                    large_cov_idxs % advantages.shape[1],
                ] = pg_losses_kl[
                    large_cov_idxs // advantages.shape[1],
                    large_cov_idxs % advantages.shape[1],
                ]

    pg_loss = reduce_loss(
        loss=pg_losses,
        loss_mask=loss_mask,
        loss_reduction=config.loss_reduction,
        max_seq_len=config.max_seq_len,
    )

    return pg_loss, {}
