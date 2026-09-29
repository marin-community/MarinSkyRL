"""Regression tests for `normalize_token_ids` and the chat-template slicing sites that depend on it.

Two production RL failures motivated these tests:

* Qwen3-Next-80B terminal_bench training: the prompt-assembly site passed a `BatchEncoding` into the
  training batch. `BatchEncoding` is a `UserDict`, not a `dict`, so iterating it yields its keys and the batch
  verifier rejected the string 'input_ids' as a token id.
* Qwen3-Next-80B no-op learning: `get_generation_prompt_ids` and `encode_messages_subset` slice an
  `apply_chat_template` result by `len()`. With a `BatchEncoding`, `len()` is the key count, so every response
  slice was empty, every trajectory was loss-masked, and training steps did nothing.
"""

import pytest
import torch
from transformers import BatchEncoding

from skyrl_train.trajectory_runners.trajectory_processing import (
    encode_messages_subset,
    get_generation_prompt_ids,
    normalize_token_ids,
)

SYSTEM_BLOCK = [100, 101]
USER_BLOCK = [200, 201, 202]
GENERATION_BLOCK = [300, 301]
ASSISTANT_BLOCK = [400, 401, 402, 403]
ROLE_BLOCKS = {"system": SYSTEM_BLOCK, "user": USER_BLOCK, "assistant": ASSISTANT_BLOCK}


def encode_conversation(messages, add_generation_prompt):
    ids = [token for message in messages for token in ROLE_BLOCKS[message["role"]]]
    if add_generation_prompt:
        ids += GENERATION_BLOCK
    return ids


class FlatListTokenizer:
    """Tokenizer whose `apply_chat_template` returns a flat token list."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, chat_template=None):
        return encode_conversation(messages, add_generation_prompt)


class BatchEncodingTokenizer:
    """Tokenizer whose `apply_chat_template` returns a `BatchEncoding`, as Qwen3-Next's bundled template does."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, chat_template=None):
        ids = encode_conversation(messages, add_generation_prompt)
        return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})


@pytest.mark.parametrize(
    "encoded",
    [
        BatchEncoding({"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]}),
        BatchEncoding({"input_ids": torch.tensor([5, 6, 7])}),
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]},
        {"token_ids": [5, 6, 7]},
        {"ids": [5, 6, 7]},
        [[5, 6, 7]],
    ],
    ids=["batch-encoding", "batch-encoding-tensor", "dict", "token-ids-key", "ids-key", "singleton-batch"],
)
def test_normalize_token_ids_flattens_encodings_to_int_list(encoded):
    normalized = normalize_token_ids(encoded)

    assert normalized == [5, 6, 7]
    assert all(type(token) is int for token in normalized)


def test_normalize_token_ids_rejects_mapping_without_token_ids():
    with pytest.raises(ValueError):
        normalize_token_ids({"attention_mask": [1, 1, 1]})


@pytest.mark.parametrize("tokenizer", [FlatListTokenizer(), BatchEncodingTokenizer()], ids=["flat", "batch-encoding"])
def test_generation_prompt_ids_are_the_template_suffix(tokenizer):
    assert get_generation_prompt_ids(tokenizer) == GENERATION_BLOCK


@pytest.mark.parametrize("tokenizer", [FlatListTokenizer(), BatchEncodingTokenizer()], ids=["flat", "batch-encoding"])
def test_encode_messages_subset_slices_out_the_assistant_block(tokenizer):
    assert encode_messages_subset([{"role": "assistant", "content": "ok"}], tokenizer) == ASSISTANT_BLOCK
