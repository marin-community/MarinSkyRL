"""CPU tests for the hardened TIS rollout-logprob alignment.

Covers the robust two-tier alignment added to fix silent TIS misalignment:
  1. EXACT path: zip vLLM logprobs onto training tokens by token id (no guessing).
  2. LCS fallback: last-resort string match, ALWAYS recorded in AlignmentStats so
     it surfaces as tis/lcs_fallback_fraction instead of silently degrading TIS.

Run:
    uv run --isolated --group dev --extra cpu pytest tests/cpu/trajectory_runners/test_tis_alignment.py
"""

import math
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from skyrl_train.group_admission import AdmissionRejection, GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.trajectory_runners.trajectory_processing import (
    AlignmentStats,
    align_logprobs_by_token_ids,
    align_logprobs_with_lcs,
    extract_logprobs_from_rollout_details,
    extract_token_ids_from_rollout_details,
    extract_prompt_token_ids_from_rollout_details,
    get_generation_prompt_ids,
    get_response_ids_and_loss_mask_from_messages,
    _tito_full_enabled,
)


def _generated_ids(tokenizer, content):
    generation_prompt_ids = get_generation_prompt_ids(tokenizer)
    response_ids = get_response_ids_and_loss_mask_from_messages([{"role": "assistant", "content": content}], tokenizer)[
        0
    ]
    generated_ids = response_ids[len(generation_prompt_ids) :]
    if tokenizer.eos_token_id in generated_ids:
        last_eos = len(generated_ids) - 1 - generated_ids[::-1].index(tokenizer.eos_token_id)
        generated_ids = generated_ids[: last_eos + 1]
    return generated_ids


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
    m = stats.as_metrics(prefix="tis/", lcs_alert_threshold=0.005)
    assert math.isclose(m["tis/exact_match_fraction"], 0.8)
    assert math.isclose(m["tis/lcs_fallback_fraction"], 0.1)
    assert math.isclose(m["tis/unaligned_fraction"], 0.1)
    # Metered LCS guard: 0.1 fallback fraction is above the 0.005 default -> alert trips.
    assert m["tis/lcs_fallback_alert"] == 1.0


def test_lcs_fallback_alert_metric():
    # No LCS -> no alert; always-present (keyset-stable) key.
    clean = AlignmentStats()
    clean.n_tokens = 100
    clean.n_exact = 100
    mc = clean.as_metrics(lcs_alert_threshold=0.005)
    assert "tis/lcs_fallback_alert" in mc and mc["tis/lcs_fallback_alert"] == 0.0
    # A tiny sub-threshold LCS fraction does NOT alert.
    tiny = AlignmentStats()
    tiny.n_tokens = 1000
    tiny.n_exact = 998
    tiny.n_lcs = 2  # 0.002 < 0.005
    assert tiny.as_metrics(lcs_alert_threshold=0.005)["tis/lcs_fallback_alert"] == 0.0
    # Above threshold -> alert.
    bad = AlignmentStats()
    bad.n_tokens = 100
    bad.n_exact = 90
    bad.n_lcs = 10  # 0.10 > 0.005
    assert bad.as_metrics(lcs_alert_threshold=0.005)["tis/lcs_fallback_alert"] == 1.0
    # Typed threshold raises the bar.
    assert bad.as_metrics(lcs_alert_threshold=0.2)["tis/lcs_fallback_alert"] == 0.0

    # Unaligned tokens are unsafe at any fraction, even when LCS was not used.
    unaligned = AlignmentStats()
    unaligned.n_tokens = 1000
    unaligned.n_exact = 999
    unaligned.n_unaligned = 1
    metrics = unaligned.as_metrics(lcs_alert_threshold=0.2)
    assert metrics["tis/lcs_fallback_alert"] == 0.0
    assert metrics["tis/alignment_alert"] == 1.0


def test_extract_float_format_no_longer_disables_tis():
    rd = [{"logprobs": [[-0.1, -0.2]], "completion_token_ids": [[10, 20]]}]
    assert extract_logprobs_from_rollout_details(rd) == [[-0.1, -0.2]]
    assert extract_token_ids_from_rollout_details(rd) == [[10, 20]]


# ---------------------------------------------------------------------------
# Full TITO: prompt-id extractor + flag scaffold (Stage 2)
# ---------------------------------------------------------------------------
def test_extract_prompt_token_ids():
    rd = [
        {
            "prompt_token_ids": [[1, 2, 3], [1, 2, 3, 10, 20, 4, 5]],
            "completion_token_ids": [[10, 20], [30]],
            "logprobs": [[-0.1, -0.2], [-0.3]],
        }
    ]
    assert extract_prompt_token_ids_from_rollout_details(rd) == [[1, 2, 3], [1, 2, 3, 10, 20, 4, 5]]
    # absent -> None (None-safe)
    assert extract_prompt_token_ids_from_rollout_details([{"completion_token_ids": [[10]]}]) is None
    assert extract_prompt_token_ids_from_rollout_details(None) is None
    assert extract_prompt_token_ids_from_rollout_details([]) is None


def test_tito_full_resolution_precedence():
    # AUTO / unset: follow whether the selected objective consumes rollout logprobs.
    assert _tito_full_enabled() is False
    assert _tito_full_enabled(rollout_logprobs_required=True) is True
    assert _tito_full_enabled(rollout_logprobs_required=False) is False

    # Explicit config overrides the objective when non-None.
    assert _tito_full_enabled(rollout_logprobs_required=True, tito_full=False) is False
    assert _tito_full_enabled(rollout_logprobs_required=False, tito_full=True) is True
    assert _tito_full_enabled(rollout_logprobs_required=True, tito_full=None) is True


def test_tito_full_without_rollout_logprob_consumer_preserves_existing_assembly():
    assert _tito_full_enabled(rollout_logprobs_required=False, tito_full=None) is False


def test_tito_assembly_declines_on_inconsistent_stream(monkeypatch):
    """When the served prompt-id stream violates the prefix invariant, the TITO
    assembly must DECLINE (return None) so the public function falls back to the
    re-tok + splice path — never silently assemble a wrong sequence."""
    from skyrl_train.trajectory_runners.trajectory_processing import _assemble_response_ids_tito_full

    class _Tok:
        eos_token_id = 999

    # prompt[1] does NOT start with prompt[0] + completion[0] -> invariant fails.
    out = _assemble_response_ids_tito_full(
        messages=[
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "c"},
        ],
        tokenizer=_Tok(),
        generation_prompt_ids=[1, 2],
        assistant_logprobs=[[-0.1], [-0.2]],
        assistant_token_ids=[[10], [30]],
        assistant_prompt_token_ids=[[1, 2], [7, 8, 9]],  # bad prefix
        assistant_routed_experts=None,
        alignment_stats=AlignmentStats(),
        custom_chat_template=None,
        chat_template_kwargs=None,
    )
    assert out is None


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
    from skyrl_train.trajectory_runners.trajectory_processing import detect_qwen3_5_empty_think_prefix

    tok = _FakeTok(think_open=900, think_close=901)
    # <|im_start|>(1) assistant(2) \n(3) <think>(900) \n\n(4) </think>(901) \n\n(5)
    gp = [1, 2, 3, 900, 4, 901, 5]
    prefix = detect_qwen3_5_empty_think_prefix(tok, gp)
    # Real prefix is everything BEFORE the injected empty <think> block.
    assert prefix == [1, 2, 3]


def test_detect_qwen3_5_empty_think_prefix_negative_dense_qwen3():
    from skyrl_train.trajectory_runners.trajectory_processing import detect_qwen3_5_empty_think_prefix

    # Dense Qwen3 gen-prompt has no think tokens at all -> None (byte-identical path).
    tok = _FakeTok(think_open=None, think_close=None)
    assert detect_qwen3_5_empty_think_prefix(tok, [1, 2, 3]) is None
    # Think tokens exist in vocab but NOT injected into the gen prompt -> None.
    tok2 = _FakeTok(think_open=900, think_close=901)
    assert detect_qwen3_5_empty_think_prefix(tok2, [1, 2, 3]) is None


def test_detect_qwen3_5_rejects_nonempty_think_block():
    from skyrl_train.trajectory_runners.trajectory_processing import detect_qwen3_5_empty_think_prefix

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
    assistant_full = get_response_ids_and_loss_mask_from_messages([messages[1]], tokenizer)[0]
    # generated tokens = full assistant encoding minus the generation-prompt prefix,
    # up to and including EOS.
    body = assistant_full[len(gen_prompt) :]
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


def test_valid_multi_turn_full_tito_preserves_all_training_logprobs():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    messages = [
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "A tool returned more evidence."},
        {"role": "assistant", "content": "Revised answer."},
    ]
    completions = [_generated_ids(tokenizer, "First answer."), _generated_ids(tokenizer, "Revised answer.")]
    generation_prompt_ids = get_generation_prompt_ids(tokenizer)
    first_prompt = [700, 701] + generation_prompt_ids
    second_prompt = first_prompt + completions[0] + [800, 801] + generation_prompt_ids
    behavior_logprobs = [[-0.1] * len(completions[0]), [-0.2] * len(completions[1])]
    stats = AlignmentStats()

    _, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        messages,
        tokenizer,
        assistant_logprobs=behavior_logprobs,
        assistant_token_ids=completions,
        assistant_prompt_token_ids=[first_prompt, second_prompt],
        rollout_logprobs_required=True,
        alignment_stats=stats,
    )

    assert [logprob for logprob, mask in zip(rollout_logprobs, loss_mask, strict=True) if mask] == sum(
        behavior_logprobs, []
    )
    assert stats.n_exact == sum(map(len, completions))
    assert stats.n_unaligned == 0


@pytest.mark.parametrize("truncate_logprobs", [False, True])
def test_failed_multi_turn_full_tito_cannot_enter_required_logprob_training(truncate_logprobs):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    messages = [
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "A tool returned more evidence."},
        {"role": "assistant", "content": "Revised answer."},
    ]
    retokenized_completions = [
        _generated_ids(tokenizer, "First answer."),
        _generated_ids(tokenizer, "Revised answer."),
    ]
    divergent_completions = [[token_id + 1 for token_id in turn] for turn in retokenized_completions]
    behavior_logprobs = [[-0.1] * (len(turn) - int(truncate_logprobs)) for turn in divergent_completions]
    generation_prompt_ids = get_generation_prompt_ids(tokenizer)
    stats = AlignmentStats()

    response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        messages,
        tokenizer,
        assistant_logprobs=behavior_logprobs,
        assistant_token_ids=divergent_completions,
        assistant_prompt_token_ids=[
            [700, 701] + generation_prompt_ids,
            [999] + generation_prompt_ids,
        ],
        rollout_logprobs_required=True,
        alignment_stats=stats,
        tis_splice=False,
    )

    assert not any(loss_mask)
    assert stats.n_unaligned == sum(map(len, retokenized_completions))
    assert all(logprob == 0.0 for logprob in rollout_logprobs)

    group = SimpleNamespace(
        trajectory_batch={
            "response_ids": [response_ids],
            "loss_masks": [loss_mask],
            "rollout_logprobs": [rollout_logprobs],
        },
        earliest_model_step=0,
    )
    policy = GroupAdmissionPolicy(
        GroupAdvantageInvariant.no_group_advantage(physical_group_size=1),
        max_staleness_steps=0,
        rollout_logprobs_required=True,
    )
    decision = policy.evaluate(group, global_step=0)
    assert decision.primary_rejection is AdmissionRejection.FULLY_MASKED


def test_partial_lcs_alignment_masks_the_message_when_logprobs_are_required():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    content = "A response with several tokens."
    generated_ids = _generated_ids(tokenizer, content)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    missing_index = len(generated_ids) // 2
    partial_logprobs = [
        {"token": token, "logprob": -0.1} for index, token in enumerate(generated_tokens) if index != missing_index
    ]
    stats = AlignmentStats()

    _, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
        [{"role": "assistant", "content": content}],
        tokenizer,
        assistant_logprobs=[partial_logprobs],
        rollout_logprobs_required=True,
        alignment_stats=stats,
    )

    assert not any(loss_mask)
    assert stats.n_unaligned == 1


def test_missing_turn_logprobs_mask_only_the_affected_message():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    messages = [
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "A tool returned more evidence."},
        {"role": "assistant", "content": "Revised answer."},
    ]
    completions = [_generated_ids(tokenizer, "First answer."), _generated_ids(tokenizer, "Revised answer.")]
    first_logprobs = [-0.1] * len(completions[0])
    stats = AlignmentStats()

    _, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
        messages,
        tokenizer,
        assistant_logprobs=[first_logprobs],
        assistant_token_ids=completions,
        rollout_logprobs_required=True,
        alignment_stats=stats,
    )

    assert [logprob for logprob, mask in zip(rollout_logprobs, loss_mask, strict=True) if mask] == first_logprobs
    assert sum(loss_mask) == len(completions[0])
    assert stats.n_unaligned == len(completions[1])


def test_float_format_without_ids_uses_positional_exact():
    """Float logprobs + matching count but no token ids -> positional 1:1 (exact)."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
    ]
    gen_prompt = get_generation_prompt_ids(tokenizer)
    assistant_full = get_response_ids_and_loss_mask_from_messages([messages[1]], tokenizer)[0]
    body = assistant_full[len(gen_prompt) :]
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
