"""Stage B (F5) — the channel threads a KNOWN per-token reward correctly.

Proves (device-agnostic; the advantage math runs on CPU): a known per-token
shaping vector survives collate->pad into TrainingInputBatch["token_level_shaping"]
at the exact token positions + magnitudes, a Stage-C-shape combiner lands the
shaping on the right tokens, and with shaping=zeros the combined advantage is
byte-identical to pure RLOO-N (the outcome term is preserved).

Run:
    pytest tests/cpu/reward/test_token_channel_threads.py
"""

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.utils.ppo_utils import compute_rloo_n_outcome_advantage


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def test_channel_lands_at_exact_positions(tokenizer):
    prompts = tokenizer(["abc", "12345"])["input_ids"]
    responses = tokenizer(["def", "67890"])["input_ids"]
    rewards = [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])]
    loss_masks = [[1, 1, 1], [1, 1, 1, 1, 1]]
    # KNOWN per-token shaping: distinct values at distinct positions.
    shaping = [[0.1, 0.2, 0.3], [0.0, -0.5, 0.0, 0.7, 0.0]]
    span_tags = [[1, 2, 3], [0, 1, 2, 3, 0]]

    out = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, rewards, loss_masks, None, None, shaping, span_tags
    )
    tls = out[7]
    rst = out[8]
    assert tls is not None and rst is not None
    # response_len padded to max(3,5)=5.
    assert tls.shape == (2, 5)
    assert rst.shape == (2, 5)
    # Exact magnitudes at exact positions (right-padded with zeros).
    assert torch.allclose(tls[0], torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0]))
    assert torch.allclose(tls[1], torch.tensor([0.0, -0.5, 0.0, 0.7, 0.0]))
    assert rst[0].tolist() == [1, 2, 3, 0, 0]
    assert rst[1].tolist() == [0, 1, 2, 3, 0]
    assert rst.dtype == torch.long


def _combine_advantage_with_shaping(advantages, token_level_shaping, response_mask):
    """The Stage-C-shape combiner (Stage B only proves the seam): ADD the
    per-token shaping onto the outcome advantage, masked to response tokens.
    Stage C registers the real PBS estimator; this is the additive contract."""
    return advantages + token_level_shaping * response_mask


def test_zeros_channel_is_pure_rloo_n():
    """channel=zeros => combined advantage byte-identical to pure RLOO-N."""
    torch.manual_seed(0)
    bsz, seqlen = 4, 6
    # Outcome reward at the last token of each trajectory (the production layout).
    token_level_rewards = torch.zeros(bsz, seqlen)
    token_level_rewards[:, -1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    response_mask = torch.ones(bsz, seqlen)
    index = np.array(["g0", "g0", "g1", "g1"])

    adv, _ = compute_rloo_n_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=type("C", (), {"rloo_n_min_group_size": 2, "rloo_n_filter_zero_reward_groups": False})(),
    )
    zeros = torch.zeros_like(adv)
    combined = _combine_advantage_with_shaping(adv, zeros, response_mask)
    # No-op: combining a zeros channel leaves RLOO-N untouched, exactly.
    assert torch.equal(combined, adv)


def test_nonzero_channel_shifts_exact_tokens():
    """A known non-zero channel adds to the advantage at exactly its token
    positions and nowhere else; the outcome term is otherwise unchanged."""
    torch.manual_seed(0)
    bsz, seqlen = 4, 6
    token_level_rewards = torch.zeros(bsz, seqlen)
    token_level_rewards[:, -1] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    response_mask = torch.ones(bsz, seqlen)
    index = np.array(["g0", "g0", "g1", "g1"])
    adv, _ = compute_rloo_n_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=type("C", (), {"rloo_n_min_group_size": 2, "rloo_n_filter_zero_reward_groups": False})(),
    )
    shaping = torch.zeros(bsz, seqlen)
    shaping[0, 2] = 0.3  # one edit token on sample 0
    shaping[2, 4] = -0.1
    combined = _combine_advantage_with_shaping(adv, shaping, response_mask)
    delta = combined - adv
    assert torch.isclose(delta[0, 2], torch.tensor(0.3))
    assert torch.isclose(delta[2, 4], torch.tensor(-0.1))
    # Everywhere else the advantage is unchanged.
    mask = torch.ones_like(delta, dtype=torch.bool)
    mask[0, 2] = False
    mask[2, 4] = False
    assert torch.allclose(delta[mask], torch.zeros_like(delta[mask]))
