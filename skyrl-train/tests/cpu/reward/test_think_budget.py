"""Think-token loss down-weighting: weighted mask values and weighted-mean denominators."""

import math

import torch
from omegaconf import OmegaConf

from skyrl_train.config.objective_spec import LossReduction
from skyrl_train.objective.reduction import policy_data_weights, reduce_to_step, step_counts
from skyrl_train.objective.losses import PolicyLossInputs, ppo_policy_loss
from skyrl_train.tensor_math import masked_mean

# Span-tag constants (mirror skyrl_train.utils.span_tagger; duplicated here to
# avoid importing the generator stack span_tagger pulls in at module load).
SPAN_OTHER, SPAN_THINK, SPAN_ACTION, SPAN_EDIT = 0, 1, 2, 3


def _loss_cfg():
    return OmegaConf.create(
        {
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "max_seq_len": 8,
        }
    )


def test_policy_loss_byte_identical_at_weight_one():
    """The full ppo_policy_loss value is bit-identical whether we pass the raw
    loss_mask or the weight=1.0 'weighted' mask (which is the same object)."""
    torch.manual_seed(0)
    B, A = 3, 5
    log_probs = torch.randn(B, A)
    old_log_probs = torch.randn(B, A)
    advantages = torch.randn(B, A)
    loss_mask = torch.tensor([[1, 1, 0, 1, 1], [1, 1, 1, 0, 0], [1, 0, 1, 1, 1]], dtype=torch.long)
    tags = torch.tensor(
        [
            [SPAN_THINK, SPAN_THINK, 0, SPAN_ACTION, SPAN_ACTION],
            [SPAN_THINK, SPAN_ACTION, SPAN_EDIT, 0, 0],
            [SPAN_ACTION, 0, SPAN_THINK, SPAN_EDIT, SPAN_ACTION],
        ],
        dtype=torch.long,
    )
    cfg = _loss_cfg()

    reference = ppo_policy_loss(PolicyLossInputs(log_probs, old_log_probs, None, advantages, loss_mask), cfg)
    wmask = policy_data_weights(loss_mask, tags, think_token_weight=1.0)
    actual = ppo_policy_loss(PolicyLossInputs(log_probs, old_log_probs, None, advantages, wmask), cfg)

    assert torch.equal(reference.values, actual.values), "weight=1.0 loss must be byte-identical"
    assert reference.metrics == actual.metrics


def test_weighted_mask_downweights_think_only():
    loss_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    #          OTHER  THINK  THINK  ACTION
    tags = torch.tensor([[SPAN_ACTION, SPAN_THINK, SPAN_THINK, SPAN_ACTION]], dtype=torch.long)
    w = 0.3
    wmask = policy_data_weights(loss_mask, tags, think_token_weight=w)
    assert wmask is not loss_mask
    expected = torch.tensor([[1.0, w, w, 1.0]])
    assert torch.allclose(wmask, expected)
    # Non-THINK positions are exactly the original mask value.
    assert wmask[0, 0].item() == 1.0 and wmask[0, 3].item() == 1.0


def test_weighted_mask_respects_zero_loss_mask():
    """A THINK token that was loss_mask==0 stays 0 after weighting (0 * w == 0)."""
    loss_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
    tags = torch.tensor([[SPAN_THINK, SPAN_THINK, SPAN_ACTION]], dtype=torch.long)
    wmask = policy_data_weights(loss_mask, tags, think_token_weight=0.5)
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
    wmask = policy_data_weights(loss_mask, tags, think_token_weight=w)
    got = masked_mean(loss, wmask)
    # weighted mean = sum(loss * weight) / sum(weight)
    num = 2.0 * 1 + 4.0 * w + 6.0 * w + 8.0 * 1
    den = 1 + w + w + 1
    assert math.isclose(got.item(), num / den, rel_tol=1e-6)


def test_weighted_reduce_loss_token_mean_matches_weighted_mean():
    loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    loss_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    tags = torch.tensor(
        [[SPAN_THINK, SPAN_ACTION, SPAN_ACTION], [SPAN_THINK, SPAN_THINK, SPAN_OTHER]], dtype=torch.long
    )
    w = 0.25
    wmask = policy_data_weights(loss_mask, tags, think_token_weight=w)
    counts = step_counts([wmask], [loss_mask], [], [torch.ones_like(wmask)], 8, lambda value: value)
    got = reduce_to_step(
        loss,
        wmask,
        counts.policy,
        LossReduction.TOKEN_MEAN,
        max_seq_len=8,
        nonzero_advantage_rows=counts.nonzero_advantage_rows,
    )
    # token_mean == masked_mean over all tokens with the weighted mask.
    num = (1.0 * w + 2.0 + 3.0) + (4.0 * w + 5.0 * w + 0.0)
    den = (w + 1 + 1) + (w + w + 0)
    assert math.isclose(got.item(), num / den, rel_tol=1e-6)
