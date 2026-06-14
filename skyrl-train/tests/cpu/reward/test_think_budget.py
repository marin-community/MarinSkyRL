"""Stage D (F7 + M4) — think-token budget unit gates.

Covers:
  1. Flag-off byte-identical (think_token_weight=1.0, think_token_cost=0.0):
     - build_think_weighted_loss_mask returns the SAME loss_mask object (so the
       load-bearing policy-loss path is bit-identical to today).
     - the policy-loss (ppo_policy_loss / reduce_loss) value is bit-identical
       whether or not span tags are supplied, at weight 1.0.
     - compute_think_token_cost with cost 0.0 returns an all-zeros (or == base)
       channel ⇒ the additive seam is a no-op.
  2. F7 weighted mask: THINK tokens get weight < 1; the weighted-mean denominator
     is correct (weighted numerator / weighted denominator); non-THINK unaffected.
  3. M4 think-cost: negative value lands on THINK tokens only; per-turn cap;
     additive onto a base (PBS) channel without disturbing EDIT entries; zeros
     when off.

Run:
    pytest tests/cpu/reward/test_think_budget.py
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.utils.ppo_utils import (
    build_think_weighted_loss_mask,
    ppo_policy_loss,
    reduce_loss,
    masked_mean,
)
from skyrl_train.utils.think_budget import compute_think_token_cost

# Span-tag constants (mirror skyrl_train.utils.span_tagger; duplicated here to
# avoid importing the generator stack span_tagger pulls in at module load).
SPAN_OTHER, SPAN_THINK, SPAN_ACTION, SPAN_EDIT = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Gate 1 — flag-off byte-identical (the critical shared-loss-path gate)
# ---------------------------------------------------------------------------
def test_weighted_mask_weight_one_returns_same_object():
    loss_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.long)
    tags = torch.tensor([[SPAN_THINK, SPAN_THINK, 0, SPAN_ACTION],
                         [SPAN_THINK, 0, SPAN_EDIT, SPAN_ACTION]], dtype=torch.long)
    out = build_think_weighted_loss_mask(loss_mask, tags, think_token_weight=1.0)
    assert out is loss_mask, "weight==1.0 must return the ORIGINAL loss_mask object"


def test_weighted_mask_no_tags_returns_same_object():
    loss_mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.long)
    out = build_think_weighted_loss_mask(loss_mask, None, think_token_weight=0.3)
    assert out is loss_mask, "absent span tags must return the ORIGINAL loss_mask object"


def _loss_cfg():
    return OmegaConf.create({
        "policy_loss_type": "regular",
        "loss_reduction": "token_mean",
        "eps_clip_low": 0.2,
        "eps_clip_high": 0.2,
        "clip_ratio_c": 3.0,
        "use_tis": False,
        "tis_imp_ratio_cap": -1.0,
        "max_seq_len": 8,
    })


def test_policy_loss_byte_identical_at_weight_one():
    """The full ppo_policy_loss value is bit-identical whether we pass the raw
    loss_mask or the weight=1.0 'weighted' mask (which is the same object)."""
    torch.manual_seed(0)
    B, A = 3, 5
    log_probs = torch.randn(B, A)
    old_log_probs = torch.randn(B, A)
    advantages = torch.randn(B, A)
    loss_mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 0, 0], [1, 0, 1, 1, 1]], dtype=torch.long)
    tags = torch.tensor([[SPAN_THINK, SPAN_THINK, 0, SPAN_ACTION, SPAN_ACTION],
                         [SPAN_THINK, SPAN_ACTION, SPAN_EDIT, 0, 0],
                         [SPAN_ACTION, 0, SPAN_THINK, SPAN_EDIT, SPAN_ACTION]], dtype=torch.long)
    cfg = _loss_cfg()

    loss_ref, clip_ref = ppo_policy_loss(log_probs, old_log_probs, advantages, cfg, loss_mask=loss_mask)
    wmask = build_think_weighted_loss_mask(loss_mask, tags, think_token_weight=1.0)
    loss_d, clip_d = ppo_policy_loss(log_probs, old_log_probs, advantages, cfg, loss_mask=wmask)

    assert torch.equal(loss_ref, loss_d), "weight=1.0 loss must be byte-identical"
    assert clip_ref == clip_d


def test_think_cost_zero_is_noop():
    tags = [SPAN_OTHER, SPAN_THINK, SPAN_THINK, SPAN_ACTION, SPAN_EDIT]
    out = compute_think_token_cost(tags, think_token_cost=0.0)
    assert out == [0.0] * len(tags)
    # And with a base channel: returns the base unchanged.
    base = [0.0, 0.0, 0.0, 0.0, 0.2]  # a pretend PBS EDIT credit
    out_b = compute_think_token_cost(tags, think_token_cost=0.0, base=base)
    assert out_b == base


# ---------------------------------------------------------------------------
# Gate 2 — F7 weighted mask: think down-weight + correct denominator
# ---------------------------------------------------------------------------
def test_weighted_mask_downweights_think_only():
    loss_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    #          OTHER  THINK  THINK  ACTION
    tags = torch.tensor([[SPAN_ACTION, SPAN_THINK, SPAN_THINK, SPAN_ACTION]], dtype=torch.long)
    w = 0.3
    wmask = build_think_weighted_loss_mask(loss_mask, tags, think_token_weight=w)
    assert wmask is not loss_mask
    expected = torch.tensor([[1.0, w, w, 1.0]])
    assert torch.allclose(wmask, expected)
    # Non-THINK positions are exactly the original mask value.
    assert wmask[0, 0].item() == 1.0 and wmask[0, 3].item() == 1.0


def test_weighted_mask_respects_zero_loss_mask():
    """A THINK token that was loss_mask==0 stays 0 after weighting (0 * w == 0)."""
    loss_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
    tags = torch.tensor([[SPAN_THINK, SPAN_THINK, SPAN_ACTION]], dtype=torch.long)
    wmask = build_think_weighted_loss_mask(loss_mask, tags, think_token_weight=0.5)
    assert wmask[0, 0].item() == 0.0  # masked-out think token stays masked out
    assert math.isclose(wmask[0, 1].item(), 0.5)
    assert wmask[0, 2].item() == 1.0


def test_weighted_mean_denominator_is_weighted():
    """masked_mean with the weighted mask = weighted numerator / weighted denom.
    Verify against a hand-computed weighted mean (this is the denominator the
    seqnorm bugs broke)."""
    loss = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    loss_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    tags = torch.tensor([[SPAN_ACTION, SPAN_THINK, SPAN_THINK, SPAN_ACTION]], dtype=torch.long)
    w = 0.5
    wmask = build_think_weighted_loss_mask(loss_mask, tags, think_token_weight=w)
    got = masked_mean(loss, wmask)
    # weighted mean = sum(loss * weight) / sum(weight)
    num = 2.0 * 1 + 4.0 * w + 6.0 * w + 8.0 * 1
    den = 1 + w + w + 1
    assert math.isclose(got.item(), num / den, rel_tol=1e-6)


def test_weighted_reduce_loss_token_mean_matches_weighted_mean():
    loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    loss_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    tags = torch.tensor([[SPAN_THINK, SPAN_ACTION, SPAN_ACTION],
                         [SPAN_THINK, SPAN_THINK, SPAN_OTHER]], dtype=torch.long)
    w = 0.25
    wmask = build_think_weighted_loss_mask(loss_mask, tags, think_token_weight=w)
    got = reduce_loss(loss, wmask, "token_mean", max_seq_len=8)
    # token_mean == masked_mean over all tokens with the weighted mask.
    num = (1.0 * w + 2.0 + 3.0) + (4.0 * w + 5.0 * w + 0.0)
    den = (w + 1 + 1) + (w + w + 0)
    assert math.isclose(got.item(), num / den, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Gate 3 — M4 think-cost: lands on THINK only, capped per turn, additive
# ---------------------------------------------------------------------------
def test_think_cost_lands_on_think_only():
    #      OTHER THINK THINK ACTION EDIT
    tags = [SPAN_OTHER, SPAN_THINK, SPAN_THINK, SPAN_ACTION, SPAN_EDIT]
    cost = 1e-3
    out = compute_think_token_cost(tags, think_token_cost=cost, max_cost_per_turn=1.0)
    # 2 think tokens, total = 2 * 1e-3 = 2e-3 (< cap), split uniformly = -1e-3 each.
    assert out[0] == 0.0
    assert math.isclose(out[1], -cost)
    assert math.isclose(out[2], -cost)
    assert out[3] == 0.0 and out[4] == 0.0


def test_think_cost_negative_sign():
    tags = [SPAN_THINK, SPAN_THINK]
    out = compute_think_token_cost(tags, think_token_cost=0.01, max_cost_per_turn=1.0)
    assert all(v <= 0.0 for v in out)
    # Even if a positive magnitude is passed, the written value is negative.
    out2 = compute_think_token_cost(tags, think_token_cost=-0.01, max_cost_per_turn=1.0)
    assert out == out2  # magnitude interpretation: sign of input is ignored


def test_think_cost_per_turn_cap():
    """A long think span has its TOTAL cost capped, so per-token cost shrinks."""
    tags = [SPAN_THINK] * 100  # one long ramble
    cost = 1e-3  # uncapped total would be 0.1
    cap = 0.05
    out = compute_think_token_cost(tags, think_token_cost=cost, max_cost_per_turn=cap)
    total = sum(out)
    assert math.isclose(total, -cap, rel_tol=1e-6), "per-turn total must be capped"
    # Uniform scatter.
    assert all(math.isclose(v, -cap / 100) for v in out)


def test_think_cost_per_turn_cap_is_per_turn_not_global():
    """Two separate think turns each get their own cap (not a single global cap)."""
    # turn 1: 60 THINK ; ACTION break ; turn 2: 60 THINK
    tags = [SPAN_THINK] * 60 + [SPAN_ACTION] + [SPAN_THINK] * 60
    cost = 1e-3
    cap = 0.05
    out = compute_think_token_cost(tags, think_token_cost=cost, max_cost_per_turn=cap)
    total = sum(out)
    # Each turn capped at 0.05 -> total ~ -0.10 (two turns).
    assert math.isclose(total, -2 * cap, rel_tol=1e-6)
    assert out[60] == 0.0  # the ACTION break carries no cost


def test_think_cost_additive_onto_pbs_base_non_overlapping():
    """M4 sums onto a Stage-C PBS base; PBS EDIT entries are preserved and THINK
    entries get the cost — non-overlapping spans, one channel."""
    tags = [SPAN_THINK, SPAN_THINK, SPAN_EDIT, SPAN_EDIT]
    base = [0.0, 0.0, 0.15, 0.15]  # PBS credit on the EDIT tokens
    cost = 1e-3
    out = compute_think_token_cost(tags, think_token_cost=cost, max_cost_per_turn=1.0, base=base)
    # EDIT entries untouched.
    assert math.isclose(out[2], 0.15) and math.isclose(out[3], 0.15)
    # THINK entries carry the (negative) cost.
    assert math.isclose(out[0], -cost) and math.isclose(out[1], -cost)


def test_think_cost_no_think_tokens_is_base():
    tags = [SPAN_ACTION, SPAN_EDIT, SPAN_OTHER]
    base = [0.0, 0.2, 0.0]
    out = compute_think_token_cost(tags, think_token_cost=1e-3, max_cost_per_turn=1.0, base=base)
    assert out == base  # nothing to charge


def test_think_cost_empty_tags():
    assert compute_think_token_cost([], think_token_cost=1e-3) == []
