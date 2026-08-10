"""Shared tensor math for policy optimization.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from jaxtyping import Float
from omegaconf import DictConfig

from skyrl_train.training_batch import TrainingInputBatch


LOG_PROB_DELTA_CLIP = 20.0


def right_pad_to_match(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Right-truncate or zero-pad the last dimension to match a reference tensor."""
    if tensor.shape == reference.shape and (dtype is None or tensor.dtype == dtype):
        return tensor
    width = min(tensor.shape[-1], reference.shape[-1])
    aligned = torch.zeros_like(reference, dtype=dtype or tensor.dtype)
    aligned[..., :width] = tensor[..., :width]
    return aligned


def safe_exp_delta(delta: torch.Tensor, clip: float = LOG_PROB_DELTA_CLIP, out_dtype=None) -> torch.Tensor:
    """Exponentiate a bounded log-probability delta without low-precision overflow."""
    result = torch.exp(delta.float().clamp(min=-clip, max=clip))
    return result.to(out_dtype or delta.dtype)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)


@torch.no_grad()
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    kl_estimator_type: str = "k3",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        loss_mask: Mask for tokens included in the loss.
        kl_estimator_type: Approximation to use: ``k1``, ``abs``, ``k2``, or ``k3``.
    """
    if kl_estimator_type == "k1":
        kld = log_probs - log_probs_base
    elif kl_estimator_type == "abs":
        kld = (log_probs - log_probs_base).abs()
    elif kl_estimator_type == "k2":
        kld = 0.5 * (log_probs - log_probs_base).square()
    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html.
    elif kl_estimator_type == "k3":
        kl = log_probs_base - log_probs
        # For numerical stability
        kl = torch.clamp(kl, min=-LOG_PROB_DELTA_CLIP, max=LOG_PROB_DELTA_CLIP)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        kld = torch.clamp(kld, min=-10, max=10)
    else:
        raise ValueError(f"Invalid KL estimator type: {kl_estimator_type}")

    if loss_mask is not None:
        kld = kld * loss_mask
    return kld


@torch.no_grad()
def normalize_advantages_dict(data: TrainingInputBatch) -> TrainingInputBatch:
    """Normalizes the advantages in the data batch.

    Expects:
        - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
        - `["response_mask"]`: Float[torch.Tensor, "batch_size seqlen"]
    """
    advantages: Float[torch.Tensor, "batch_size seqlen"] = data["advantages"]
    response_masks: Float[torch.Tensor, "batch_size seqlen"] = data["response_mask"]
    num_actions: float = response_masks.sum()
    mean: float = advantages.mean()
    variance_numerator: float = ((advantages - mean).pow(2) * response_masks).sum()
    rstd: float = (variance_numerator / num_actions).clamp(min=1e-8).rsqrt()

    data["advantages"] = (advantages - mean) * rstd
    return data


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def ppo_critic_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[float]]:
    if config.value_clip is not None:
        values_clipped = old_values + (values - old_values).clamp(-config.value_clip, config.value_clip)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
        clipfrac = masked_mean((surr1 > surr2).float(), loss_mask).mean().detach().item()
    else:
        clipfrac = None
        loss = (values - returns) ** 2

    loss = masked_mean(loss, loss_mask, dim=-1).mean()
    return 0.5 * loss, clipfrac
