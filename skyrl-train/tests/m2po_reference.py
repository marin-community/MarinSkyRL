"""Verbatim M2PO released threshold functions, kept independent of production.

Source: https://github.com/Infini-AI-Lab/M2PO/blob/af54a3e8feffb7a66a8258003aebefa37759cec0/verl/trainer/ppo/core_algos.py
Apache-2.0, original M2PO/verl authors. Formatting is normalized by repository lint.
These functions retain the reference's discrete threshold and diagnostic edge cases.
"""

import torch


def _solve_tau_from_sorted_delta2(sorted_delta2: torch.Tensor, target_sum: float) -> float:
    """
    Given sorted ascending values v_i = Δ_i^2 (i=0..n-1) and a target sum S,
    find τ^2 such that sum_i min(v_i, τ^2) = S.
    This uses a single pass over breakpoints without binary search.

    Returns:
        tau (float): sqrt(τ^2). If S >= sum(v_i), returns +inf (no clipping needed).
                     If S <= 0, returns 0.0 (clip everything to 0).
    """

    if sorted_delta2.numel() == 0:
        return 100000

    total = float(sorted_delta2.sum().item())
    if target_sum >= total - 1e-12:  # no clipping needed
        return 100000
    if target_sum <= 1e-12:  # clip everything to 0
        return 0.0

    csum = torch.cumsum(sorted_delta2, dim=0)  # prefix sums
    n = sorted_delta2.numel()

    for k in range(0, n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12
        if m2 * rest + left_sum >= target_sum - 1e-12:
            # print(f"================")
            # print(f"n: {n}, k: {k}, left_sum: {left_sum}, target_sum: {target_sum}")
            # print(f"sorted_delta2[k]: {sorted_delta2[k].item()}")
            # print(f"{list(zip(sorted_delta2[k-5:k+5].tolist(), csum[k-5:k+5].tolist()))}")
            # print((sorted_delta2 == 0).float().mean())
            # print(f"{target_sum}")
            if k == 0:
                return 0.0, csum[-1].item() / n
            else:
                M2_after = (sorted_delta2[k - 1].item() * (rest + 1) + float(csum[k - 1].item())) / n
                return float(sorted_delta2[k - 1].item() - 1e-12) ** 0.5, M2_after

    return 100000


def _get_trust_region_tokens_delta_sq(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
):
    mask = response_mask.bool()
    adv_example = advantages[:, 0]
    pos_adv_mask = adv_example > 1e-12
    neg_adv_mask = adv_example < -1e-12

    delta = old_log_prob - log_prob  # Δ = log p_old - log p_new
    ratio = torch.exp(-delta)  # r = exp(log_new - log_old)

    pos_adv_response_mask = mask[pos_adv_mask]
    neg_adv_response_mask = mask[neg_adv_mask]

    pos_adv_ratio = ratio[pos_adv_mask]
    neg_adv_ratio = ratio[neg_adv_mask]

    pos_adv_r_gt_1_mask = pos_adv_ratio > 1.0 + 1e-12
    neg_adv_r_lt_1_mask = neg_adv_ratio < 1.0 - 1e-12

    delta_sq = delta.pow(2)
    pos_adv_harm_tokens_delta_sq = delta_sq[pos_adv_mask][pos_adv_r_gt_1_mask & pos_adv_response_mask]
    neg_adv_harm_tokens_delta_sq = delta_sq[neg_adv_mask][neg_adv_r_lt_1_mask & neg_adv_response_mask]

    tr_tokens_delta_sq = torch.cat([pos_adv_harm_tokens_delta_sq, neg_adv_harm_tokens_delta_sq])

    return tr_tokens_delta_sq


def kpo_clip_harmful_tokens(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    KL2_budget: float = None,
):
    """
    Decide global clip scalars (clip_low, clip_high) under an M2 budget.

    Policy:
      - Consider only harmful tokens: (A>0 & r>1) or (A<0 & r<1), where r = exp(log_new - log_old).
      - Sort harmful tokens by delta^2 = (log p_old - log p_new)^2 ascending.
      - Find a single threshold τ so that capping |delta| at τ across harmful tokens
        yields overall M2 <= KL2_budget.
      - Map τ to two global ratio bounds:
            clip_low  = exp(-τ)  (applies to adv<0 & r<1)
            clip_high = exp(+τ)  (applies to adv>0 & r>1)
      - Non-harmful quadrants are not constrained by these bounds.

    Returns:
      clip_low  (float): lower clamp for tokens with (adv<0 & r<1)
      clip_high (float): upper clamp for tokens with (adv>0 & r>1)
    """
    assert KL2_budget is not None, "KL2_budget must be set."

    tr_tokens_delta_sq = _get_trust_region_tokens_delta_sq(old_log_prob, log_prob, advantages, response_mask)
    token_num = tr_tokens_delta_sq.numel()

    if token_num == 0:  # no clipping needed
        return 0.0, 100000, 0.0, 0.0

    target_total = KL2_budget * float(token_num)
    M2_now = float(tr_tokens_delta_sq.sum().detach().item() / token_num)

    if M2_now <= KL2_budget + 1e-12:
        # No clipping needed -> effectively no constraint
        return 0.0, 100000, M2_now, M2_now

    print(f"tr-M2_now: {M2_now}")
    print(f"KL2_budget: {KL2_budget}")

    # import pdb; pdb.set_trace()

    sorted_delta2, _ = torch.sort(tr_tokens_delta_sq)  # ascending
    tau, M2_after = _solve_tau_from_sorted_delta2(sorted_delta2, target_total)

    # Map |Δ|<=τ to ratio bounds per quadrant
    clip_low = float(torch.exp(torch.tensor(-tau)).item())  # applies to (adv<0, r<1)
    clip_high = float(torch.exp(torch.tensor(+tau)).item())  # applies to (adv>0, r>1)

    return clip_low, clip_high, M2_now, M2_after


# The released loss body below is unchanged. Only unused ratio diagnostics are
# stubbed; token-mean reduction is implemented independently for this CPU oracle.
class verl_F:
    @staticmethod
    def masked_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp_min(1)


def agg_loss(loss_mat, loss_mask, loss_agg_mode):
    assert loss_agg_mode == "token-mean"
    return verl_F.masked_mean(loss_mat, loss_mask)


def get_ratio_stats(*args):
    return {}


def compute_m2po_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    M2_budget: float = None,
    miniclip_low: float = 0.3,
    miniclip_high: float = 0.5,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute policy loss under an M2 (KL^2) budget using per-token clipping bounds.

    Steps:
      1) Get per-token (clip_low, clip_high) from kpo_clip.
      2) Compute ratio and apply element-wise clamp.
      3) Compute surrogate loss -A * ratio_clipped and aggregate.

    Returns:
      pg_loss:       aggregated policy loss
      stats:         dict with basic diagnostics (M2 before/after, fractions)
      clip_low/high: the per-token bounds actually used
    """

    clip_low, clip_high, M2_data, M2_after = kpo_clip_harmful_tokens(
        old_log_prob, log_prob, advantages, response_mask, M2_budget
    )

    clip_low = 1 - clip_low
    clip_high = clip_high - 1
    print(f"clip_low: {clip_low}, clip_high: {clip_high}")
    if miniclip_low is not None and clip_low < miniclip_low:
        clip_low = miniclip_low
    if miniclip_high is not None and clip_high < miniclip_high:
        clip_high = miniclip_high

    # ratio = exp(log_new - log_old)
    ratio = torch.exp(log_prob - old_log_prob)
    ppo_kl = verl_F.masked_mean(-(log_prob - old_log_prob), response_mask)

    ratio_stats = get_ratio_stats(ratio, advantages, response_mask, log_prob, old_log_prob)

    ##### clip
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - clip_low, 1 + clip_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_loss = agg_loss(loss_mat=clip_pg_losses1, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    ratio_stats["m2po/clip_low"] = clip_low
    ratio_stats["m2po/clip_high"] = clip_high
    ratio_stats["m2po/M2"] = M2_data
    ratio_stats["m2po/M2_after"] = M2_after
    ratio_stats["m2po/M2_budget"] = M2_budget

    return pg_loss, pg_clipfrac, ppo_kl, (ppo_kl - ppo_kl), ratio_stats
