"""
uv run --isolated --group dev --extra cpu pytest tests/cpu/dataset/test_preprocess.py
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from skyrl_train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
)


@pytest.fixture
def cfg():
    return OmegaConf.create({"trainer": {"max_prompt_length": 10}, "generator": {"max_generate_length": 5}})


class CharTokenizer:
    """Character-level tokenizer: "abc" -> [97, 98, 99]. The tests below hardcode these ids."""

    pad_token_id = 0

    def __call__(self, texts):
        return {"input_ids": [[ord(c) for c in text] for text in texts]}


@pytest.fixture
def tokenizer():
    return CharTokenizer()


def test_convert_prompts_responses_to_batch_tensors_exact(tokenizer, cfg):
    prompts = ["abc", "12345"]
    outputs = ["def", "67890"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]

    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]
    rewards = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0, 0, 0])]

    (
        sequences,
        attention_mask,
        action_mask,
        ret_rewards,
        ret_loss_masks,
        ret_log_probs,
        ret_routed_experts,
        _tls,
        _rst,
    ) = convert_prompts_responses_to_batch_tensors(
        tokenizer,
        prompts,
        outputs,
        rewards,
        loss_masks,
    )

    # loss mask should be the same length as the action mask (padded to the longest input)
    assert sequences.shape[0] == len(prompts)
    assert action_mask.shape == ret_loss_masks.shape
    assert torch.equal(ret_loss_masks[0], torch.tensor([1, 1, 0, 0, 0]))
    assert torch.equal(ret_loss_masks[1], torch.tensor([1, 1, 1, 0, 0]))
    assert torch.equal(ret_rewards[0], torch.tensor([0, 1, 0, 0, 0]))
    assert torch.equal(ret_rewards[1], torch.tensor([1, 0, 0, 0, 0]))


def test_convert_prompts_responses_to_batch_tensors_different_lengths(cfg, tokenizer):
    # Test with inputs of different lengths
    prompts = ["Short", "This is a longer prompt"]
    outputs = ["Long response here", "Short"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]
    rewards = [torch.tensor([1.0, 0.5, 0.3]), torch.tensor([0.8])]
    loss_masks = [[1, 1, 1], [1]]

    (
        sequences,
        attention_mask,
        action_mask,
        ret_rewards,
        ret_loss_masks,
        ret_log_probs,
        ret_routed_experts,
        _tls,
        _rst,
    ) = convert_prompts_responses_to_batch_tensors(
        tokenizer,
        prompts,
        outputs,
        rewards,
        loss_masks,
    )

    max_response_len = max([len(output) for output in outputs])

    # Check shapes
    assert sequences.shape[0] == 2  # batch size
    assert attention_mask.shape == sequences.shape
    # Tensor.shape can be directly compared with tuples
    assert action_mask.shape == (2, max_response_len)
    assert ret_rewards.shape == (2, max_response_len)
    assert ret_loss_masks.shape == (2, max_response_len)

    # Verify padding is applied correctly
    # First input is shorter than second input. the input is left padded
    assert sequences[0, 0] == tokenizer.pad_token_id
    # second output is shorter than first output. the output is right padded
    assert sequences[1, -1] == tokenizer.pad_token_id


def _re_inputs(tokenizer):
    """Two samples, response lens 3 and 5 (matches the _exact fixture shapes)."""
    prompts = tokenizer(["abc", "12345"])["input_ids"]
    outputs = tokenizer(["def", "67890"])["input_ids"]
    rewards = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])]
    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]
    return prompts, outputs, rewards, loss_masks


def test_routed_experts_uniform_LK(tokenizer, cfg):
    # All per-token rows are a canonical [L=48, K=8] vector. Result must be
    # [batch, max_response_len, 48, 8] with sentinel-zero rows in the padding.
    prompts, outputs, rewards, loss_masks = _re_inputs(tokenizer)
    L, K = 48, 8
    row = [[1] * K for _ in range(L)]
    # sample 0 has 3 generated rows, sample 1 has 5.
    routed_experts = [np.asarray([row] * 3, dtype=np.uint8), np.asarray([row] * 5, dtype=np.uint8)]

    out = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, outputs, rewards, loss_masks, None, routed_experts
    )
    re_tensor = out[6]
    assert re_tensor is not None
    assert re_tensor.shape == (2, 5, L, K)
    # sample 0 padding rows (positions 3,4) are sentinel zeros
    assert torch.equal(re_tensor[0, 3], torch.zeros(L, K, dtype=torch.uint8))
    assert torch.equal(re_tensor[0, 0], torch.ones(L, K, dtype=torch.uint8))


def test_routed_experts_mixed_sentinel_and_real_no_ragged_crash(tokenizer, cfg):
    # A missing sample may have narrower route geometry than a captured sibling.
    prompts, outputs, rewards, loss_masks = _re_inputs(tokenizer)
    L, K = 48, 8
    real_row = [[2] * K for _ in range(L)]
    routed_experts = [
        np.asarray([real_row] * 3, dtype=np.uint8),
        np.zeros((5, 1, 1), dtype=np.uint8),
    ]

    out = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, outputs, rewards, loss_masks, None, routed_experts
    )
    re_tensor = out[6]
    assert re_tensor is not None
    assert re_tensor.shape == (2, 5, L, K)
    assert torch.equal(re_tensor[1], torch.zeros(5, L, K, dtype=torch.uint8))
    assert torch.equal(re_tensor[0, 0], torch.full((L, K), 2, dtype=torch.uint8))
