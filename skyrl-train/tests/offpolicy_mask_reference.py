"""Scalar oracles independent of the production vectorized mask implementation.

Band semantics match verl d040717b's rollout-correction integration anchor;
the row veto is the separately specified INTELLECT-3 threshold. The M2 mask
enumerates every harmful prefix and does not reuse production tensor logic.
"""

import math

import torch


def offpolicy_keep_reference(delta, selected, low=0.5, high=5.0, veto=1e-5):
    keep = []
    for row, valid in zip(delta.detach().cpu().tolist(), selected.cpu().tolist(), strict=True):
        ratios = [math.exp(max(-20.0, min(20.0, value))) for value in row]
        vetoed = any(flag and ratio < veto for ratio, flag in zip(ratios, valid, strict=True))
        keep.append(
            [bool(flag and not vetoed and low <= ratio <= high) for ratio, flag in zip(ratios, valid, strict=True)]
        )
    return torch.tensor(keep, dtype=torch.bool, device=delta.device)


def minimal_m2_reference(delta, advantages, selected, tau):
    values = delta.detach().flatten().cpu().tolist()
    adv = advantages.flatten().cpu().tolist()
    support = selected.flatten().cpu().tolist()
    candidates = [
        i
        for i, (value, advantage, valid) in enumerate(zip(values, adv, support, strict=True))
        if valid and value * advantage > 0
    ]
    candidates.sort(key=lambda i: values[i] ** 2, reverse=True)
    removed = set()
    satisfied = False
    for count in range(len(candidates) + 1):
        removed = set(candidates[:count])
        retained = [value**2 for i, value in enumerate(values) if support[i] and i not in removed]
        if retained and sum(retained) / len(retained) < tau:
            satisfied = True
            break
    return torch.tensor([i in removed for i in range(len(values))], device=delta.device).reshape_as(selected), satisfied


def regular_correction_reference_policy_loss(
    log_probs,
    old_log_probs,
    advantages,
    config,
    loss_mask=None,
    rollout_logprobs=None,
    global_loss_denom=None,
    *,
    mode,
):
    """Independent mask and analytic PPO derivative for the native actor fixture."""
    from tests.tis_reference import regular_tis_scalar_reference

    assert loss_mask is not None and rollout_logprobs is not None
    assert config.loss_reduction == "token_mean" and global_loss_denom is None
    selected = loss_mask > 0
    if mode == "offpolicy":
        keep = offpolicy_keep_reference(old_log_probs - rollout_logprobs, selected)
    elif mode == "m2":
        removed, _ = minimal_m2_reference(log_probs - old_log_probs, advantages, selected, 0.04)
        keep = selected & ~removed
    else:
        raise ValueError(mode)
    value, derivative = regular_tis_scalar_reference(
        log_probs,
        old_log_probs,
        old_log_probs,
        advantages * keep,
        loss_mask,
        low=float(config.eps_clip_low),
        high=float(config.eps_clip_high),
        cap=1.0,
    )
    tangent = ((log_probs - log_probs.detach()) * derivative.to(log_probs)).sum()
    return log_probs.new_tensor(value) + tangent, {}
