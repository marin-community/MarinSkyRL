"""Policy loss implementations and clipping diagnostics.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Optional, Protocol

import loguru
import torch
from omegaconf import DictConfig

from skyrl_train.utils.importance_ratio_diagnostics import compute_tis_diagnostics
from skyrl_train.utils.loss_reduction import (
    GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION,
    SEQUENCE_MEAN_LOSS_REDUCTION,
    SUPPORTED_LOSS_REDUCTIONS,
    LossReduction,
    build_think_weighted_loss_mask,
    reduce_loss,
)
from skyrl_train.utils.algorithm_registry import PolicyLossType, register_policy_loss
from skyrl_train.utils.policy_math import LOG_PROB_DELTA_CLIP, compute_approx_kl, masked_mean, safe_exp_delta


@dataclass(frozen=True)
class PolicyClipMetrics:
    """Stable clipping diagnostic record emitted by every training backend."""

    ppo_clip_ratio: float = 0.0
    ppo_clip_ratio_low: float = 0.0
    ppo_clip_ratio_high: float = 0.0
    ppo_clip_pressure_low: float = 0.0
    ppo_clip_pressure_high: float = 0.0
    ppo_ratio_exact_unit_fraction: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


POLICY_CLIP_METRIC_KEYS = tuple(field.name for field in fields(PolicyClipMetrics))


class LossScaling(StrEnum):
    """Which layer divides an objective across accumulated microbatches."""

    CALLER = "caller"
    MEGATRON_PIPELINE = "megatron_pipeline"


@dataclass(frozen=True)
class PolicyObjective:
    """Backend-neutral policy objective and its observable components."""

    optimization_loss: torch.Tensor
    unscaled_loss: torch.Tensor
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    kl_loss: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class PolicyAuxiliaryTerms:
    entropy: torch.Tensor
    kl_loss: torch.Tensor
    loss: torch.Tensor


class PolicyLossFunction(Protocol):
    """Callable contract implemented by registered policy losses."""

    def __call__(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        *,
        config: DictConfig,
        loss_mask: Optional[torch.Tensor],
        rollout_logprobs: Optional[torch.Tensor],
        global_loss_denom: Optional[float],
    ) -> tuple[torch.Tensor, dict[str, float]]: ...


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
    """Report clipping decisions, pre-clamp pressure, and exact-unit ratios.

    The low/high clip ratios count selected tokens beyond the matching bound;
    pressure counts every token beyond each bound before objective selection.
    The pooled ratio counts all selected tokens unless the caller supplies its
    own loss-specific ``pooled_clip_ratio``. The exact-unit fraction distinguishes
    a genuinely inert clipping geometry from missing diagnostics.
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
        ppo_ratio_exact_unit_fraction=_masked_fraction(ratio == 1, loss_mask),
    ).as_dict()


def complete_clip_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Fill omitted clipping diagnostics at the training-backend boundary."""
    return PolicyClipMetrics().as_dict() | metrics


def _compute_policy_auxiliary_terms(
    *,
    action_log_probs: torch.Tensor,
    base_action_log_probs: Optional[torch.Tensor],
    token_entropy: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    config: DictConfig,
) -> PolicyAuxiliaryTerms:
    entropy = masked_mean(token_entropy, loss_mask)
    entropy_term = entropy * config.entropy_loss_coef if config.use_entropy_loss else entropy.new_zeros(())

    if config.use_kl_loss:
        if base_action_log_probs is None:
            raise ValueError("base_action_log_probs are required when use_kl_loss is enabled")
        kl_loss = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            loss_mask=loss_mask,
            kl_estimator_type=config.kl_estimator_type,
        )
        kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
    else:
        kl_loss = action_log_probs.new_zeros(())
    return PolicyAuxiliaryTerms(
        entropy=entropy,
        kl_loss=kl_loss,
        loss=kl_loss * config.kl_loss_coef - entropy_term,
    )


def _scale_policy_objective(
    policy_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    *,
    accumulation_steps: int,
    scaling: LossScaling,
    loss_reduction: LossReduction,
) -> torch.Tensor:
    globally_normalized = loss_reduction == GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION
    if scaling is LossScaling.MEGATRON_PIPELINE:
        # Megatron Core divides the closure result by the number of microbatches.
        # A globally normalized policy term already spans that window, so multiply
        # it here to cancel Megatron's division while auxiliaries remain averaged.
        policy_scale = accumulation_steps if globally_normalized else 1
        return policy_loss * policy_scale + auxiliary_loss
    if scaling is LossScaling.CALLER:
        if globally_normalized:
            return policy_loss + auxiliary_loss / accumulation_steps
        return (policy_loss + auxiliary_loss) / accumulation_steps
    raise ValueError(f"Unknown loss scaling owner: {scaling!r}")


def _policy_objective_metrics(
    policy_loss_metrics: dict[str, float],
    *,
    old_action_log_probs: torch.Tensor,
    rollout_logprobs: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    config: DictConfig,
) -> dict[str, float]:
    metrics = complete_clip_metrics(policy_loss_metrics)
    if config.use_tis or config.get("policy_loss_type") == PolicyLossType.BEHAVIOR_CLIP:
        metrics.update(
            compute_tis_diagnostics(
                old_action_log_probs,
                rollout_logprobs,
                loss_mask,
                cap=config.tis_imp_ratio_cap,
            )
        )
        if not config.use_tis:
            # The disabled cap sentinel is not an applied correction threshold.
            # Config is identical on every rank, preserving collective keysets.
            del metrics["tis/imp_ratio_capped_fraction"]
    return metrics


def compute_policy_objective(
    *,
    action_log_probs: torch.Tensor,
    old_action_log_probs: torch.Tensor,
    base_action_log_probs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    rollout_logprobs: Optional[torch.Tensor],
    response_span_tags: Optional[torch.Tensor],
    token_entropy: torch.Tensor,
    config: DictConfig,
    policy_loss_fn: PolicyLossFunction,
    accumulation_steps: int,
    scaling: LossScaling,
    global_loss_denom: Optional[float] = None,
) -> PolicyObjective:
    """Build a policy objective independent of the backend execution loop.

    Backends must backpropagate ``optimization_loss``. ``unscaled_loss`` is the
    backend-independent value to report; it combines policy and auxiliary terms
    before caller- or scheduler-owned gradient-accumulation scaling.
    """
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be positive, got {accumulation_steps}")
    globally_normalized = config.loss_reduction == GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION
    if globally_normalized and global_loss_denom is None:
        raise ValueError(
            f"global_loss_denom is required for {GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION}"
        )

    policy_loss_mask = build_think_weighted_loss_mask(
        loss_mask,
        response_span_tags,
        float(config.think_token_weight),
    )
    policy_loss, policy_loss_metrics = policy_loss_fn(
        action_log_probs,
        old_action_log_probs,
        advantages,
        config=config,
        loss_mask=policy_loss_mask,
        rollout_logprobs=rollout_logprobs,
        global_loss_denom=global_loss_denom,
    )

    auxiliary = _compute_policy_auxiliary_terms(
        action_log_probs=action_log_probs,
        base_action_log_probs=base_action_log_probs,
        token_entropy=token_entropy,
        loss_mask=loss_mask,
        config=config,
    )
    unscaled_loss = policy_loss + auxiliary.loss
    optimization_loss = _scale_policy_objective(
        policy_loss,
        auxiliary.loss,
        accumulation_steps=accumulation_steps,
        scaling=scaling,
        loss_reduction=config.loss_reduction,
    )
    metrics = _policy_objective_metrics(
        policy_loss_metrics,
        old_action_log_probs=old_action_log_probs,
        rollout_logprobs=rollout_logprobs,
        loss_mask=loss_mask,
        config=config,
    )
    return PolicyObjective(
        optimization_loss=optimization_loss,
        unscaled_loss=unscaled_loss,
        policy_loss=policy_loss,
        entropy=auxiliary.entropy,
        kl_loss=auxiliary.kl_loss,
        metrics=metrics,
    )


@register_policy_loss(PolicyLossType.REGULAR)
@register_policy_loss(PolicyLossType.DUAL_CLIP)
def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    global_loss_denom: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    assert config.policy_loss_type in ["regular", "dual_clip"], "loss_type must be either 'regular' or 'dual_clip'"
    loss_reduction = config.loss_reduction
    assert loss_reduction in SUPPORTED_LOSS_REDUCTIONS, (
        f"loss_reduction must be one of {list(SUPPORTED_LOSS_REDUCTIONS)}"
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
        global_denom=global_loss_denom,
    )
    return loss, clip_metrics


@register_policy_loss(PolicyLossType.BEHAVIOR_CLIP)
def behavior_clipped_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    global_loss_denom: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pessimistic PPO clipping against the policy that generated each token.

    Adapted from AReaL's ``areal.utils.functional.ppo_actor_loss_fn``.
    """
    del old_log_probs
    if rollout_logprobs is None:
        raise ValueError("rollout_logprobs are required for behavior_clip policy loss")
    if config.use_tis:
        raise ValueError("behavior_clip cannot be combined with use_tis; both correct against rollout_logprobs")
    if config.loss_reduction not in SUPPORTED_LOSS_REDUCTIONS:
        raise ValueError(f"loss_reduction must be one of {list(SUPPORTED_LOSS_REDUCTIONS)}")

    ratio = safe_exp_delta(log_probs - rollout_logprobs, out_dtype=log_probs.dtype)
    clipped_ratio = torch.clamp(ratio, 1.0 - config.eps_clip_low, 1.0 + config.eps_clip_high)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    loss = torch.maximum(pg_loss1, pg_loss2)
    clip_metrics = clipping_metrics(
        ratio,
        pg_loss1.detach() < pg_loss2.detach(),
        loss_mask,
        eps_clip_low=config.eps_clip_low,
        eps_clip_high=config.eps_clip_high,
    )

    pg_loss3 = torch.sign(advantages) * config.clip_ratio_c * advantages
    loss = torch.where(advantages < 0, torch.minimum(loss, pg_loss3), loss)
    loss = reduce_loss(
        loss,
        loss_mask,
        config.loss_reduction,
        config.max_seq_len,
        global_denom=global_loss_denom,
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
    global_loss_denom: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    SAPO (Soft Adaptive Policy Optimization) policy loss function.

    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    """
    # SAPO has only been established as stable with sequence_mean reduction.
    loss_reduction = config.loss_reduction
    if loss_reduction != SEQUENCE_MEAN_LOSS_REDUCTION:
        loguru.logger.warning(
            f"With SAPO it's recommended to use '{SEQUENCE_MEAN_LOSS_REDUCTION}' loss reduction; got {loss_reduction}"
        )

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
    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len, global_denom=global_loss_denom)

    return loss, {}


@register_policy_loss(PolicyLossType.GSPO)
def gspo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    global_loss_denom: Optional[float] = None,
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
    loss_reduction = config.loss_reduction

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

    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len, global_denom=global_loss_denom)

    return loss, clip_metrics


@register_policy_loss(PolicyLossType.CISPO)
def compute_policy_loss_cispo(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    global_loss_denom: Optional[float] = None,
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

    loss = reduce_loss(
        loss,
        loss_mask,
        config.loss_reduction,
        config.max_seq_len,
        global_denom=global_loss_denom,
    )
    return loss, clip_metrics


@register_policy_loss(PolicyLossType.CLIP_COV)
def compute_policy_loss_clip_cov(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    global_loss_denom: Optional[float] = None,
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
        global_denom=global_loss_denom,
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
    global_loss_denom: Optional[float] = None,
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
        global_denom=global_loss_denom,
    )

    return pg_loss, {}
