"""The vectorised collator must match the per-row reference bitwise on every shape."""

import random
from unittest.mock import MagicMock

import pytest
import torch

from skyrl_train.dataset.preprocess import (
    collate_response_token_channel,
    convert_prompts_responses_to_batch_tensors,
)

PAD_TOKEN_ID = 0


def _reference_collate(tokenizer, prompts, responses, rewards, loss_masks, logprobs):
    """The per-row Python-list collator this module replaced, kept verbatim as the oracle."""
    max_input_len, max_output_len = 0, 0
    prompt_token_lens, response_token_lens = [], []
    inputs_token_ids, outputs_token_ids = [], []
    for prompt, response in zip(prompts, responses):
        inputs_token_ids.append(prompt)
        outputs_token_ids.append(response)
        prompt_token_len = len(prompt)
        response_token_len = len(response)
        prompt_token_lens.append(prompt_token_len)
        response_token_lens.append(response_token_len)
        max_input_len = max(max_input_len, prompt_token_len)
        max_output_len = max(max_output_len, response_token_len)

    pad_token_id = tokenizer.pad_token_id
    sequences = []
    attention_masks = []
    action_masks = []
    for i, prompt in enumerate(prompts):
        input_len = prompt_token_lens[i]
        input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])
        input_attention_mask = [0] * (max_input_len - input_len) + [1] * input_len
        output_len = response_token_lens[i]
        output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)
        output_attention_mask = [1] * output_len + [0] * (max_output_len - output_len)
        sequences.append(input_ids + output_ids)
        attention_masks.append(input_attention_mask + output_attention_mask)
        action_masks.append(output_attention_mask)

    sequences = torch.tensor(sequences)
    attention_mask = torch.tensor(attention_masks, dtype=torch.int64)
    action_mask = torch.tensor(action_masks, dtype=torch.int64)

    ret_loss_masks = torch.zeros_like(action_mask, dtype=torch.float)
    for i, loss_mask in enumerate(loss_masks):
        ret_loss_masks[i, : len(loss_mask)] = torch.tensor(loss_mask)

    ret_rewards = torch.zeros_like(action_mask, dtype=torch.float)
    for i, custom_reward in enumerate(rewards):
        if isinstance(custom_reward, list):
            custom_reward = torch.tensor(custom_reward)
        ret_rewards[i, : len(custom_reward)] = custom_reward

    logprobs_tensor = None
    if logprobs:
        max_output_len = action_mask.size(1)
        padded_logprobs = [
            sample_logprobs + [0.0] * (max_output_len - len(sample_logprobs)) for sample_logprobs in logprobs
        ]
        logprobs_tensor = torch.tensor(padded_logprobs, dtype=torch.float)
    return sequences, attention_mask, action_mask, ret_rewards, ret_loss_masks, logprobs_tensor


def _reference_channel(rows, response_template, dtype):
    result = torch.zeros_like(response_template, dtype=dtype)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row, dtype=dtype)
        result[index, : len(values)] = values
    return result


@pytest.fixture
def tokenizer():
    mock = MagicMock()
    mock.pad_token_id = PAD_TOKEN_ID
    return mock


def _assert_same(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def _random_lengths(rng, batch, kind, minimum=0):
    if kind == "same":
        length = rng.randint(minimum, 7)
        return [length] * batch
    if kind == "max":
        return [7] * batch
    if kind == "empty":
        return [0] * batch
    return [rng.randint(minimum, 7) for _ in range(batch)]


@pytest.mark.parametrize("seed", range(40))
def test_collator_matches_reference_bitwise(tokenizer, seed):
    rng = random.Random(seed)
    batch = rng.randint(1, 6)
    # A prompt always carries tokens; responses may be empty, all alike or at the maximum.
    prompt_lengths = _random_lengths(rng, batch, rng.choice(["random", "same", "max"]), minimum=1)
    response_lengths = _random_lengths(rng, batch, rng.choice(["random", "same", "max", "empty", "random"]))
    prompts = [[rng.randint(1, 200_000) for _ in range(length)] for length in prompt_lengths]
    responses = [[rng.randint(1, 200_000) for _ in range(length)] for length in response_lengths]
    loss_masks = [[rng.randint(0, 1) for _ in range(length)] for length in response_lengths]
    reward_kind = rng.choice(["float_list", "int_list", "tensor", "double_tensor"])
    rewards = []
    for length in response_lengths:
        values = [rng.uniform(-3, 3) for _ in range(length)]
        if reward_kind == "float_list":
            rewards.append(values)
        elif reward_kind == "int_list":
            rewards.append([rng.randint(-2, 2) for _ in range(length)])
        elif reward_kind == "tensor":
            rewards.append(torch.tensor(values))
        else:
            rewards.append(torch.tensor(values, dtype=torch.float64))
    logprobs = None
    if rng.random() < 0.6:
        logprobs = [[rng.uniform(-20, 0) for _ in range(length)] for length in response_lengths]

    expected = _reference_collate(tokenizer, prompts, responses, rewards, loss_masks, logprobs)
    actual = convert_prompts_responses_to_batch_tensors(tokenizer, prompts, responses, rewards, loss_masks, logprobs)

    assert actual[6] is None and actual[7] is None and actual[8] is None
    for actual_tensor, expected_tensor in zip(actual[:6], expected, strict=True):
        _assert_same(actual_tensor, expected_tensor)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("dtype", [torch.float, torch.long])
def test_response_token_channel_matches_reference_bitwise(seed, dtype):
    rng = random.Random(seed)
    batch = rng.randint(1, 5)
    lengths = _random_lengths(rng, batch, rng.choice(["random", "same", "empty"]))
    width = max(lengths) + rng.randint(0, 2)
    template = torch.zeros(batch, width, dtype=torch.int64)
    if dtype is torch.float:
        rows = [[rng.uniform(-2, 2) for _ in range(length)] for length in lengths]
    else:
        rows = [[rng.randint(0, 3) for _ in range(length)] for length in lengths]

    actual = collate_response_token_channel(rows, template, dtype=dtype, expected_lengths=lengths)

    _assert_same(actual, _reference_channel(rows, template, dtype))
    assert collate_response_token_channel(None, template, dtype=dtype, expected_lengths=lengths) is None


@pytest.mark.parametrize("bad_token", ["7", 7.0])
def test_non_int_token_id_names_the_offending_element(tokenizer, bad_token):
    prompts = [[1, 2], [3, bad_token, 5]]
    responses = [[6], [7]]

    with pytest.raises(
        ValueError, match=f"prompt token-id list at sample index 1 contains a non-int element {bad_token!r}"
    ):
        convert_prompts_responses_to_batch_tensors(tokenizer, prompts, responses, [[1.0], [0.0]], [[1], [1]])
