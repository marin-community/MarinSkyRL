"""CPU tests for the hardened TIS rollout-logprob alignment.

Covers the robust two-tier alignment added to fix silent TIS misalignment:
  1. EXACT path: zip vLLM logprobs onto training tokens by token id (no guessing).
  2. LCS fallback: last-resort string match, ALWAYS recorded in AlignmentStats so
     it surfaces as tis/lcs_fallback_fraction instead of silently degrading TIS.

Run:
    uv run --isolated --extra dev pytest tests/cpu/generators/test_tis_alignment.py
"""

import math

import pytest
from transformers import AutoTokenizer

from skyrl_train.generators.utils import (
    AlignmentStats,
    align_logprobs_by_token_ids,
    align_logprobs_with_lcs,
    extract_logprobs_from_rollout_details,
    extract_token_ids_from_rollout_details,
    get_generation_prompt_ids,
    get_response_ids_and_loss_mask_from_messages,
)


# ---------------------------------------------------------------------------
# Pure-function unit tests (no tokenizer / no network)
# ---------------------------------------------------------------------------
def test_exact_path_matches_by_token_id():
    stats = AlignmentStats()
    out = align_logprobs_by_token_ids([10, 20, 30], [10, 20, 30], [-0.1, -0.2, -0.3], stats=stats)
    assert out == [-0.1, -0.2, -0.3]
    assert stats.n_exact == 3
    assert stats.n_lcs == 0


def test_exact_path_returns_none_on_id_divergence():
    # IDs diverge -> caller must fall back; exact path declines (returns None).
    assert align_logprobs_by_token_ids([10, 20, 99], [10, 20, 30], [-0.1, -0.2, -0.3]) is None
    # Parallel-array contract violation (len mismatch) -> also None.
    assert align_logprobs_by_token_ids([10, 20], [10, 20], [-0.1]) is None
    # No data -> None.
    assert align_logprobs_by_token_ids([10], None, None) is None


def test_lcs_records_fallback_in_stats():
    stats = AlignmentStats()
    # retok ids [1,2,3] map to strings via a stub tokenizer.
    class _Tok:
        def convert_ids_to_tokens(self, ids):
            return {1: "Hello", 2: " world", 3: "!"}.get
    tok = _Tok()
    tok.convert_ids_to_tokens = lambda ids: ["Hello", " world", "!"]
    vllm = [{"token": "Hello", "logprob": -0.1}, {"token": " world", "logprob": -0.2}, {"token": "!", "logprob": -0.3}]
    out = align_logprobs_with_lcs([1, 2, 3], vllm, tok, stats=stats)
    assert out == [-0.1, -0.2, -0.3]
    assert stats.n_lcs == 3
    assert stats.n_lcs_messages == 1
    assert stats.n_unaligned == 0


def test_lcs_partial_match_counts_unaligned():
    stats = AlignmentStats()
    tok = type("T", (), {"convert_ids_to_tokens": lambda self, ids: ["A", "X", "B"]})()
    vllm = [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -0.3}]
    out = align_logprobs_with_lcs([1, 2, 3], vllm, tok, stats=stats)
    # "A" and "B" match; the middle "X" has no vLLM counterpart -> 0.0 + unaligned.
    assert out[0] == -0.1 and out[2] == -0.3 and out[1] == 0.0
    assert stats.n_lcs == 2
    assert stats.n_unaligned == 1


def test_metrics_fractions():
    stats = AlignmentStats()
    stats.n_tokens = 10
    stats.n_exact = 8
    stats.n_lcs = 1
    stats.n_unaligned = 1
    stats.n_failed_messages = 0
    m = stats.as_metrics(prefix="tis/")
    assert math.isclose(m["tis/exact_match_fraction"], 0.8)
    assert math.isclose(m["tis/lcs_fallback_fraction"], 0.1)
    assert math.isclose(m["tis/unaligned_fraction"], 0.1)


def test_extract_float_format_no_longer_disables_tis():
    rd = [{"logprobs": [[-0.1, -0.2]], "completion_token_ids": [[10, 20]]}]
    assert extract_logprobs_from_rollout_details(rd) == [[-0.1, -0.2]]
    assert extract_token_ids_from_rollout_details(rd) == [[10, 20]]


# ---------------------------------------------------------------------------
# qwen3_5/3.6 empty-think prefix detection + served-id splice (arch-gated)
# ---------------------------------------------------------------------------


class _FakeTok:
    """Minimal tokenizer stub exposing only what the detection helper needs."""

    def __init__(self, think_open=None, think_close=None, unk=None):
        self._map = {"<think>": think_open, "</think>": think_close}
        self.unk_token_id = unk

    def convert_tokens_to_ids(self, tok):
        return self._map.get(tok)


def test_detect_qwen3_5_empty_think_prefix_positive():
    from skyrl_train.generators.utils import detect_qwen3_5_empty_think_prefix

    tok = _FakeTok(think_open=900, think_close=901)
    # <|im_start|>(1) assistant(2) \n(3) <think>(900) \n\n(4) </think>(901) \n\n(5)
    gp = [1, 2, 3, 900, 4, 901, 5]
    prefix = detect_qwen3_5_empty_think_prefix(tok, gp)
    # Real prefix is everything BEFORE the injected empty <think> block.
    assert prefix == [1, 2, 3]


def test_detect_qwen3_5_empty_think_prefix_negative_dense_qwen3():
    from skyrl_train.generators.utils import detect_qwen3_5_empty_think_prefix

    # Dense Qwen3 gen-prompt has no think tokens at all -> None (byte-identical path).
    tok = _FakeTok(think_open=None, think_close=None)
    assert detect_qwen3_5_empty_think_prefix(tok, [1, 2, 3]) is None
    # Think tokens exist in vocab but NOT injected into the gen prompt -> None.
    tok2 = _FakeTok(think_open=900, think_close=901)
    assert detect_qwen3_5_empty_think_prefix(tok2, [1, 2, 3]) is None


def test_detect_qwen3_5_rejects_nonempty_think_block():
    from skyrl_train.generators.utils import detect_qwen3_5_empty_think_prefix

    tok = _FakeTok(think_open=900, think_close=901)
    # <think> ... 3 content tokens ... </think>  -> NOT an empty block -> None.
    gp = [1, 2, 900, 50, 51, 52, 901, 5]
    assert detect_qwen3_5_empty_think_prefix(tok, gp) is None


# ---------------------------------------------------------------------------
# Integration: exact path through get_response_ids_and_loss_mask_from_messages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"])
def test_exact_alignment_end_to_end(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    # Re-derive EXACTLY what vLLM would have generated for the assistant turn by
    # slicing the assistant message the same way the function does, so the
    # completion_token_ids match the re-tokenized generated tokens by construction.
    gen_prompt = get_generation_prompt_ids(tokenizer)
    assistant_full = get_response_ids_and_loss_mask_from_messages(
        [messages[1]], tokenizer
    )[0]
    # generated tokens = full assistant encoding minus the generation-prompt prefix,
    # up to and including EOS.
    body = assistant_full[len(gen_prompt):]
    if tokenizer.eos_token_id in body:
        last_eos = len(body) - 1 - body[::-1].index(tokenizer.eos_token_id)
        gen_ids = body[: last_eos + 1]
    else:
        gen_ids = body
    vllm_logprobs = [-0.01 * (i + 1) for i in range(len(gen_ids))]

    stats = AlignmentStats()
    response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        messages[1:],
        tokenizer,
        assistant_logprobs=[vllm_logprobs],
        assistant_token_ids=[gen_ids],
        alignment_stats=stats,
    )
    assert len(rollout_logprobs) == len(response_ids) == len(loss_mask)
    # The exact path should have fired for all generated tokens, NO LCS fallback.
    assert stats.n_exact == len(gen_ids)
    assert stats.n_lcs == 0
    assert stats.n_lcs_messages == 0
    assert stats.n_failed_messages == 0
    # The masked (generated) positions carry the exact vLLM logprobs in order.
    masked_lps = [lp for lp, m in zip(rollout_logprobs, loss_mask) if m == 1]
    assert masked_lps == vllm_logprobs


def test_float_format_without_ids_uses_positional_exact():
    """Float logprobs + matching count but no token ids -> positional 1:1 (exact)."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
    ]
    gen_prompt = get_generation_prompt_ids(tokenizer)
    assistant_full = get_response_ids_and_loss_mask_from_messages([messages[1]], tokenizer)[0]
    body = assistant_full[len(gen_prompt):]
    if tokenizer.eos_token_id in body:
        last_eos = len(body) - 1 - body[::-1].index(tokenizer.eos_token_id)
        gen_ids = body[: last_eos + 1]
    else:
        gen_ids = body
    vllm_logprobs = [-0.05] * len(gen_ids)
    stats = AlignmentStats()
    _, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        messages[1:],
        tokenizer,
        assistant_logprobs=[vllm_logprobs],
        assistant_token_ids=None,  # no ids -> positional exact path
        alignment_stats=stats,
    )
    assert stats.n_exact == len(gen_ids)
    assert stats.n_lcs == 0
    masked_lps = [lp for lp, m in zip(rollout_logprobs, loss_mask) if m == 1]
    assert masked_lps == vllm_logprobs
