"""CPU test for `_normalize_prompt_token_ids` in the terminal_bench generator.

Regression test for the 80B Qwen3-Next production RL crash (job 631790): the
agentic terminal_bench prompt-assembly site fed a non-flat-list shape into the
training batch, which `dataset/preprocess._verify_inputs` rejected at the first
training step with:

    ValueError: prompt token-id list at sample index 0 contains a non-int
    element 'input_ids' (type str); expected a flat list of token ids.

The leaking shape was a `BatchEncoding` / dict (iterating it yields its KEYS,
so element 0 was the string 'input_ids'). The fix normalizes the
`apply_chat_template` result to a flat `List[int]` at the assembly site, where
the ids are still present.

Run:
    uv run --isolated --extra dev pytest tests/cpu/generators/test_normalize_prompt_token_ids.py
"""

import pytest

from examples.terminal_bench.terminal_bench_generator import _normalize_prompt_token_ids
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors

from unittest.mock import MagicMock


def test_flat_list_passthrough():
    """The normal/correct path (flat List[int]) is returned unchanged."""
    ids = [151644, 872, 198, 14990, 1879]
    out = _normalize_prompt_token_ids(ids)
    assert out == ids
    assert all(isinstance(t, int) for t in out)


def test_batchencoding_dict_extracts_input_ids():
    """A BatchEncoding/dict yields its 'input_ids' value, not its keys.

    This is the exact shape that crashed job 631790: without the fix, the dict
    would be passed through and iterate to ['input_ids', 'attention_mask', ...].
    """
    enc = {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}
    out = _normalize_prompt_token_ids(enc)
    assert out == [1, 2, 3, 4]
    assert all(isinstance(t, int) for t in out)


def test_batchencoding_subclass_extracts_input_ids():
    """A real BatchEncoding (dict subclass) is handled like a dict."""
    try:
        from transformers import BatchEncoding
    except Exception:  # pragma: no cover - transformers always present in dev
        pytest.skip("transformers not available")
    enc = BatchEncoding({"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]})
    out = _normalize_prompt_token_ids(enc)
    assert out == [5, 6, 7]


def test_singleton_batched_nesting_unwrapped():
    """A [[int, ...]] singleton-batched result is unwrapped to [int, ...]."""
    out = _normalize_prompt_token_ids([[10, 11, 12]])
    assert out == [10, 11, 12]


def test_alternate_id_keys():
    for key in ("token_ids", "ids"):
        out = _normalize_prompt_token_ids({key: [9, 8, 7]})
        assert out == [9, 8, 7]


def test_dict_without_id_key_raises():
    with pytest.raises(ValueError):
        _normalize_prompt_token_ids({"attention_mask": [1, 1, 1]})


def test_normalized_prompt_collates_into_training_batch():
    """End-to-end: a normalized prompt (was a dict) collates without the
    `_verify_inputs` crash; the dict shape would have raised."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    # Pre-fix: this dict would have leaked into prompts and crashed _verify_inputs.
    raw_prompt_encoding = {"input_ids": [101, 102, 103], "attention_mask": [1, 1, 1]}
    prompt_ids = _normalize_prompt_token_ids(raw_prompt_encoding)

    prompts = [prompt_ids]
    responses = [[201, 202]]
    rewards = [[0.0, 1.0]]
    loss_masks = [[1, 1]]

    seq, attn, action_mask, ret_rewards, ret_loss, lp, re_t = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer, prompts, responses, rewards, loss_masks
        )
    )
    # prompt(3) + response(2) = 5 tokens, batch of 1
    assert seq.shape == (1, 5)
