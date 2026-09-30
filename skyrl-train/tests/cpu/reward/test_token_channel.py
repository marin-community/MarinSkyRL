"""Per-token reward channel (token_level_shaping, response_span_tags) through collate and concatenation."""

from types import SimpleNamespace

import torch

from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches

# The collator only reads pad_token_id from the tokenizer.
TOKENIZER = SimpleNamespace(pad_token_id=0)

PROMPTS = [[101], [102, 103]]
RESPONSES = [[201, 202, 203], [204, 205, 206, 207, 208]]
REWARDS = [torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])]
LOSS_MASKS = [[1, 1, 1], [1, 1, 1, 1, 1]]


def test_channel_lands_at_exact_positions():
    shaping = [[0.1, 0.2, 0.3], [0.0, -0.5, 0.0, 0.7, 0.0]]
    span_tags = [[1, 2, 3], [0, 1, 2, 3, 0]]

    out = convert_prompts_responses_to_batch_tensors(
        TOKENIZER, PROMPTS, RESPONSES, REWARDS, LOSS_MASKS, None, None, shaping, span_tags
    )
    tls, rst = out[7], out[8]

    # Right-padded with zeros to the longest response.
    assert torch.allclose(tls, torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0], [0.0, -0.5, 0.0, 0.7, 0.0]]))
    assert rst.dtype == torch.long
    assert rst.tolist() == [[1, 2, 3, 0, 0], [0, 1, 2, 3, 0]]


def test_channel_absent_when_not_provided():
    out = convert_prompts_responses_to_batch_tensors(TOKENIZER, PROMPTS, RESPONSES, REWARDS, LOSS_MASKS)
    assert out[7] is None
    assert out[8] is None


def _trajectory_batch(prompt, response, **channel):
    return {
        "prompt_token_ids": [prompt],
        "response_ids": [response],
        "rewards": [[0.0] * (len(response) - 1) + [1.0]],
        "loss_masks": [[1] * len(response)],
        "stop_reasons": ["stop"],
        "rollout_logprobs": None,
        "rollout_metrics": {},
        **channel,
    }


def test_concatenate_omits_channel_keys_when_no_batch_has_them():
    out = _trajectory_batch([1, 2], [3, 4, 5])
    merged = concatenate_trajectory_batches([out, out], tis_lcs_alert_threshold=0.005)
    assert "token_level_shaping" not in merged
    assert "response_span_tags" not in merged
    assert "rollout_routed_experts" not in merged


def test_concatenate_zero_fills_channel_for_batches_without_it():
    """Mixed batches stay 1:1 with response_ids: missing channels are zero-filled."""
    with_channel = _trajectory_batch(
        [1, 2], [3, 4, 5], token_level_shaping=[[0.5, 0.0, 0.0]], response_span_tags=[[1, 2, 2]]
    )
    without_channel = _trajectory_batch([9], [6, 7])
    merged = concatenate_trajectory_batches([with_channel, without_channel], tis_lcs_alert_threshold=0.005)
    assert merged["token_level_shaping"] == [[0.5, 0.0, 0.0], [0.0, 0.0]]
    assert merged["response_span_tags"] == [[1, 2, 2], [0, 0]]
