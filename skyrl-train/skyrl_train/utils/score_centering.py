"""Score centering for the PPO objective with truncated behavior importance weights.

The current trainer policy is ``p``, the stored policy at the start of the
optimizer update is ``o``, and the policy that sampled each token is ``q``.
For a sampled token, PPO contributes ``A * min(o/q, cap) * (p/o) * score(p)``
while its directional clip is inactive, and zero while it is active. The
correction subtracts the expectation of this same score coefficient under q.
All probabilities and clipping decisions in the correction are detached.
"""

from __future__ import annotations

import math

import torch

from skyrl_train.tensor_math import LOG_PROB_DELTA_CLIP


def _bounded_ratio(numerator_logprob: torch.Tensor, denominator_logprob: torch.Tensor) -> torch.Tensor:
    return (numerator_logprob - denominator_logprob).clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP).exp()


def ppo_tis_score_centering_correction(
    current_topk_logprobs: torch.Tensor,
    old_topk_logprobs: torch.Tensor,
    behavior_topk_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    tis_cap: float,
    eps_clip_low: float,
    eps_clip_high: float,
    tail_floor: float = 1e-6,
) -> torch.Tensor:
    """Return the per-token term to add before the PPO loss reduction.

    The head contains exact full-vocabulary logprobs for the same top-k token
    IDs under all three policies. Outside the head, q and o are modeled as
    rescaled copies of p, each preserving its measured tail mass. This keeps
    their ratios and PPO clip decision constant over the modeled tail.
    """
    shape = current_topk_logprobs.shape
    if (
        len(shape) != 3
        or shape[-1] == 0
        or old_topk_logprobs.shape != shape
        or behavior_topk_logprobs.shape != shape
        or advantages.shape != shape[:2]
        or loss_mask.shape != shape[:2]
    ):
        raise ValueError("score centering requires aligned [batch, response, top-k] logprobs and advantages")
    if tis_cap <= 0 or not 0 <= eps_clip_low < 1 or eps_clip_high < 0 or tail_floor <= 0:
        raise ValueError("invalid score-centering TIS, PPO clipping, or tail-floor parameter")

    # Invalid sentinel rows are allowed only where the policy loss is masked.
    # Replace them before any exp or multiply so NaN * 0 cannot poison a batch.
    valid = loss_mask.to(torch.bool).unsqueeze(-1)
    compute_dtype = torch.float64 if current_topk_logprobs.dtype == torch.float64 else torch.float32
    masked_logprob = -math.log(shape[-1] + 1)
    current = current_topk_logprobs.to(compute_dtype).masked_fill(~valid, masked_logprob)
    old = old_topk_logprobs.to(compute_dtype).masked_fill(~valid, masked_logprob)
    behavior = behavior_topk_logprobs.to(compute_dtype).masked_fill(~valid, masked_logprob)
    if not all(torch.isfinite(tensor).all() for tensor in (current, old, behavior)):
        raise ValueError("score centering requires finite head logprobs on every supplied position")

    current_mass = current.exp()
    old_mass = old.exp()
    behavior_mass = behavior.exp()
    if torch.any(current_mass.sum(dim=-1) > 1 + 1e-4) or torch.any(old_mass.sum(dim=-1) > 1 + 1e-4):
        raise ValueError("trainer top-k probabilities exceed full-vocabulary mass")
    if torch.any(behavior_mass.sum(dim=-1) > 1 + 1e-4):
        raise ValueError("behavior top-k probabilities exceed full-vocabulary mass")

    p_tail = (1 - current_mass.sum(dim=-1)).clamp_min(tail_floor)
    o_tail = (1 - old_mass.sum(dim=-1)).clamp_min(tail_floor)
    q_tail = (1 - behavior_mass.sum(dim=-1)).clamp_min(tail_floor)

    ppo_ratio = _bounded_ratio(current, old)
    tis_weight = _bounded_ratio(old, behavior).clamp(max=tis_cap)
    positive = advantages.unsqueeze(-1) >= 0
    active = torch.where(positive, ppo_ratio <= 1 + eps_clip_high, ppo_ratio >= 1 - eps_clip_low)
    head_coefficient = behavior_mass * tis_weight * ppo_ratio * active

    # On the modeled tail, old_v / p_v = o_tail / p_tail and
    # q_v / p_v = q_tail / p_tail. The resulting weighted score coefficient
    # can therefore be collapsed with sum_v p_v * score_v = 0.
    behavior_over_p = q_tail / p_tail
    tail_ppo_ratio = p_tail / o_tail
    tail_tis_weight = (o_tail / q_tail).clamp(max=tis_cap)
    tail_active = torch.where(
        advantages >= 0,
        tail_ppo_ratio <= 1 + eps_clip_high,
        tail_ppo_ratio >= 1 - eps_clip_low,
    )
    tail_coefficient = behavior_over_p * tail_tis_weight * tail_ppo_ratio * tail_active
    residual = head_coefficient - tail_coefficient.unsqueeze(-1) * current_mass
    correction = advantages * (residual.detach() * current).sum(dim=-1)
    return correction.masked_fill(~loss_mask.to(torch.bool), 0)
