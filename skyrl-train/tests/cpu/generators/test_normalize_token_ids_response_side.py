"""CPU test for `normalize_token_ids` on the RESPONSE-side `len()`-slicing sites.

Regression test for the Qwen3-Next-80B production RL no-op-learning bug: the
two response-side helpers in ``skyrl_train.generators.utils`` slice an
``apply_chat_template(..., tokenize=True)`` result with ``len()``-based slicing:

  * ``get_generation_prompt_ids``:
        empty_user_with_generation_prompt[len(empty_user):]
  * ``encode_messages_subset``:
        full_conversation_token_ids[len(base_conversation_token_ids):]

On the Qwen3-Next-80B tokenizer (transformers 4.57+) with the bundled
``qwen3_thinking_acc.jinja2`` template, ``apply_chat_template`` returns a
``transformers.BatchEncoding`` (a ``UserDict``), NOT a flat ``List[int]``.
``len(BatchEncoding) == 2`` (the KEY count), so the slices returned ``[]`` ->
empty ``response_ids`` / ``loss_mask`` for EVERY trajectory -> "All outputs are
loss masked" -> NaN advantages -> zero grad / entropy / loss -> no-op step.

The fix coerces all four ``apply_chat_template`` results through
``normalize_token_ids`` (BatchEncoding -> its ``input_ids`` value) BEFORE the
``len()``/slice, and is a byte-identical passthrough for the flat-list (8B/a3)
path.

Run:
    uv run --isolated --extra dev pytest \
        tests/cpu/generators/test_normalize_token_ids_response_side.py
"""

import pytest

from skyrl_train.generators.utils import (
    normalize_token_ids,
    get_generation_prompt_ids,
    encode_messages_subset,
)


# ---------------------------------------------------------------------------
# Fake tokenizers: one returns flat List[int] (8B/a3 path), one returns a
# transformers.BatchEncoding (the Qwen3-Next-80B bundled-template path).
# ---------------------------------------------------------------------------

# Deterministic per-message token blocks so slicing is checkable.
_SYSTEM_BLOCK = [100, 101]
_USER_BLOCK = [200, 201, 202]
_GEN_BLOCK = [300, 301]  # the generation-prompt suffix
_ASSISTANT_BLOCK = [400, 401, 402, 403]  # the response block we want sliced out


def _encode(messages, add_generation_prompt):
    """Map a conversation -> a flat list of ints (per-role blocks)."""
    ids = []
    for m in messages:
        if m["role"] == "system":
            ids += _SYSTEM_BLOCK
        elif m["role"] == "user":
            ids += _USER_BLOCK
        elif m["role"] == "assistant":
            ids += _ASSISTANT_BLOCK
    if add_generation_prompt:
        ids += _GEN_BLOCK
    return ids


class _FlatListTokenizer:
    """8B/a3-style: apply_chat_template returns a flat List[int]."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, chat_template=None):
        return _encode(messages, add_generation_prompt)


class _BatchEncodingTokenizer:
    """Qwen3-Next-80B-style: apply_chat_template returns a BatchEncoding."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, chat_template=None):
        from transformers import BatchEncoding

        ids = _encode(messages, add_generation_prompt)
        # BatchEncoding is a UserDict: len()==2 (keys), iterating yields keys.
        return BatchEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)})


# ---------------------------------------------------------------------------
# normalize_token_ids unit behavior (mirrors the prompt-side test).
# ---------------------------------------------------------------------------


def test_flat_list_passthrough_identity():
    ids = [151644, 872, 198, 14990, 1879]
    out = normalize_token_ids(ids)
    assert out == ids
    assert all(isinstance(t, int) for t in out)


def test_batchencoding_extracts_input_ids():
    from transformers import BatchEncoding

    enc = BatchEncoding({"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]})
    assert not isinstance(enc, dict)  # the trap: BatchEncoding is not a dict
    assert len(enc) == 2  # the trap: len() is the KEY count, not token count
    assert normalize_token_ids(enc) == [5, 6, 7]


def test_tensor_value_flattened():
    import torch
    from transformers import BatchEncoding

    enc = BatchEncoding({"input_ids": torch.tensor([5, 6, 7])})
    assert normalize_token_ids(enc) == [5, 6, 7]


def test_singleton_batched_unwrapped():
    assert normalize_token_ids([[10, 11, 12]]) == [10, 11, 12]


# ---------------------------------------------------------------------------
# get_generation_prompt_ids: BatchEncoding case must yield the real gen suffix,
# flat-list case must be unchanged.
# ---------------------------------------------------------------------------


def test_get_generation_prompt_ids_batchencoding_nonempty():
    """Pre-fix: len(BatchEncoding)==2 -> slice [2:] of a 2-key dict -> []."""
    out = get_generation_prompt_ids(_BatchEncodingTokenizer())
    assert out == _GEN_BLOCK
    assert len(out) > 0


def test_get_generation_prompt_ids_flat_list_unchanged():
    out = get_generation_prompt_ids(_FlatListTokenizer())
    assert out == _GEN_BLOCK


def test_get_generation_prompt_ids_both_paths_agree():
    assert get_generation_prompt_ids(_BatchEncodingTokenizer()) == get_generation_prompt_ids(_FlatListTokenizer())


# ---------------------------------------------------------------------------
# encode_messages_subset: the assistant response block must be sliced out
# non-empty under BatchEncoding (the actual no-op-learning trigger).
# ---------------------------------------------------------------------------


def test_encode_messages_subset_batchencoding_nonempty_response():
    """Pre-fix: full_conversation_token_ids[len(base):] with base a BatchEncoding
    -> [2:] of the full BatchEncoding -> [] -> empty response_ids -> all masked."""
    messages = [{"role": "assistant", "content": "ok"}]
    out = encode_messages_subset(messages, _BatchEncodingTokenizer())
    assert out == _ASSISTANT_BLOCK
    assert len(out) > 0


def test_encode_messages_subset_flat_list_unchanged():
    messages = [{"role": "assistant", "content": "ok"}]
    out = encode_messages_subset(messages, _FlatListTokenizer())
    assert out == _ASSISTANT_BLOCK


def test_encode_messages_subset_both_paths_agree():
    messages = [{"role": "assistant", "content": "ok"}]
    assert encode_messages_subset(messages, _BatchEncodingTokenizer()) == encode_messages_subset(
        messages, _FlatListTokenizer()
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
