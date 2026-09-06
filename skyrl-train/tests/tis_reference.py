"""Independent finite-fixture oracle for regular PPO with capped TIS.

No production loss/reduction helpers are used. Fixtures stay away from clipping
kinks and exponent-overflow guards. Old and behavior probabilities and advantages
are fixed observations, so only current-policy log probabilities differentiate.
"""

import math

import torch


def regular_tis_scalar_reference(
    log_probs, old_log_probs, rollout_logprobs, advantages, loss_mask, *, low=0.2, high=0.2, cap=2.0
):
    """Return a token-mean scalar and its analytic current-logprob derivative."""
    rows = [
        tensor.detach().double().cpu().flatten().tolist()
        for tensor in (log_probs, old_log_probs, rollout_logprobs, advantages, loss_mask)
    ]
    denominator = max(sum(rows[-1]), 1.0)
    numerator = 0.0
    derivatives = []
    for current, old, behavior, advantage, mask in zip(*rows, strict=True):
        assert abs(current - old) < 10 and abs(old - behavior) < 10
        ratio = math.exp(current - old)
        weight = min(math.exp(old - behavior), cap)
        clipped = (advantage > 0 and ratio > 1 + high) or (advantage < 0 and ratio < 1 - low)
        selected_ratio = ((1 + high) if advantage > 0 else (1 - low)) if clipped else ratio
        numerator -= mask * advantage * selected_ratio * weight
        derivative = 0.0 if clipped else -mask * advantage * ratio * weight / denominator
        derivatives.append(derivative)
    return numerator / denominator, torch.tensor(derivatives, dtype=torch.float64).reshape(log_probs.shape)


def regular_tis_reference_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, global_loss_denom=None
):
    """Test-only analytic gradient injected into the real worker optimizer path.

    The zero-valued tangent supplies the independently calculated derivative;
    it does not reuse production PPO, TIS, or reduction implementations.
    """
    assert config.loss_reduction == "token_mean" and global_loss_denom is None
    assert rollout_logprobs is not None and loss_mask is not None
    value, derivative = regular_tis_scalar_reference(
        log_probs,
        old_log_probs,
        rollout_logprobs,
        advantages,
        loss_mask,
        low=float(config.eps_clip_low),
        high=float(config.eps_clip_high),
        cap=float(config.tis_imp_ratio_cap),
    )
    gradient = derivative.to(device=log_probs.device, dtype=log_probs.dtype)
    tangent = ((log_probs - log_probs.detach()) * gradient).sum()
    return log_probs.new_tensor(value) + tangent, {}
