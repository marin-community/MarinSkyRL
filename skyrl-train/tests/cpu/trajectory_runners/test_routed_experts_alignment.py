"""CPU tests for the Stage 1 MoE router-replay capture rail (`routed_experts`).

These cover the SkyRL-side seams added in Stage 1 of the FSDP2 EP/router-replay
port (see notes/skyrl/stage1_capture_rail_scope.md). NO MoE math / replay logic
here — pure data-plane alignment + collation.

Run:
    uv run --isolated --group dev --extra cpu pytest tests/cpu/trajectory_runners/test_routed_experts_alignment.py
"""

import pytest
import numpy as np
import torch
from transformers import AutoTokenizer

from skyrl_train.trajectory_runners.trajectory_processing import (
    extract_routed_experts_from_rollout_details,
    align_routed_experts_with_lcs,
    get_response_ids_and_loss_mask_from_messages,
    get_generation_prompt_ids,
    encode_messages_subset,
    concatenate_trajectory_batches,
    SENTINEL_EXPERT_ID,
)
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors

from unittest.mock import MagicMock


# Synthetic per-token [L, K] routed_experts rows, small L=4, K=2.
L, K = 4, 2


def _real_row(seed):
    """A deterministic non-sentinel [L, K] row (expert ids in [1, num_experts))."""
    return [[(seed + layer * K + k) % 7 + 1 for k in range(K)] for layer in range(L)]


def _is_sentinel_row(row):
    return all(all(e == SENTINEL_EXPERT_ID for e in layer) for layer in row)


# ---------------------------------------------------------------------------
# Case 1: alignment round-trip + sentinel placement + multi-turn advance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_name",
    ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-0.6B"],
    ids=["qwen2_5", "qwen3"],
)
def test_alignment_round_trip_and_sentinels(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_prompt_ids = get_generation_prompt_ids(tokenizer)

    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good"},
    ]

    def num_generated(content):
        msg = [{"role": "assistant", "content": content}]
        ids = encode_messages_subset(msg, tokenizer)
        last_eos = len(ids) - 1 - ids[::-1].index(tokenizer.eos_token_id)
        return last_eos + 1 - len(generation_prompt_ids)

    n1 = num_generated("Hello there")
    n2 = num_generated("Good")

    # Per-turn routed_experts: real [L, K] rows, one per generated token.
    re_turn1 = [_real_row(10 + i) for i in range(n1)]
    re_turn2 = [_real_row(50 + i) for i in range(n2)]
    assistant_routed_experts = [re_turn1, re_turn2]

    out = get_response_ids_and_loss_mask_from_messages(
        messages, tokenizer, assistant_routed_experts=assistant_routed_experts
    )
    assert len(out) == 4, "with routed_experts the chokepoint must return a 4-tuple"
    response_ids, loss_mask, rollout_logprobs, routed_experts = out

    # Invariant: len(routed_experts) == len(response_ids) (scope Q3 / utils :699).
    assert len(routed_experts) == len(response_ids)
    assert len(loss_mask) == len(response_ids)
    # rollout_logprobs is None here (we passed no assistant_logprobs).
    assert rollout_logprobs is None

    # Every row is [L, K].
    for row in routed_experts:
        assert len(row) == L
        assert all(len(layer) == K for layer in row)

    # Non-sentinel rows EXACTLY cover loss_mask==1; loss_mask==0 rows are sentinel
    # (scope Q3 invariant #3).
    for m, row in zip(loss_mask, routed_experts):
        if m == 1:
            assert not _is_sentinel_row(row), "generated tokens must carry real routing"
        else:
            assert _is_sentinel_row(row), "user/prefix/post-EOS rows must be sentinel"

    # The generated rows in order must equal the concatenation of the two turns'
    # real rows (proves multi-turn assistant_msg_idx advance + correct copy).
    gen_rows = [row for m, row in zip(loss_mask, routed_experts) if m == 1]
    assert gen_rows == re_turn1 + re_turn2


# ---------------------------------------------------------------------------
# Case 2: tokenizer mismatch — vLLM split differently from retok, LCS aligns
# ---------------------------------------------------------------------------
def test_tokenizer_mismatch_lcs():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    # Re-tokenized generated ids (N tokens) and vLLM rows with a DIFFERENT count
    # (N+1, e.g. vLLM split one token into two). align_routed_experts_with_lcs
    # must still produce exactly len(retok) rows and copy real rows for the
    # positionally-matched prefix.
    retok_ids = [101, 102, 103, 104]  # N = 4
    vllm_rows = [_real_row(i) for i in range(5)]  # N+1 = 5

    aligned = align_routed_experts_with_lcs(retok_ids, vllm_rows, tokenizer)
    assert len(aligned) == len(retok_ids)
    for row in aligned:
        assert len(row) == L and all(len(layer) == K for layer in row)
    # The longest common positional run [0..3] copies the first 4 vLLM rows.
    assert aligned[:4] == vllm_rows[:4]

    # Exact 1:1 count → direct copy.
    aligned_eq = align_routed_experts_with_lcs(retok_ids, vllm_rows[:4], tokenizer)
    assert aligned_eq == vllm_rows[:4]

    # Empty vLLM rows → [] (caller sentinel-pads).
    assert align_routed_experts_with_lcs(retok_ids, [], tokenizer) == []


# ---------------------------------------------------------------------------
# extract helper: reads rollout_details[0]["extra"]["routed_experts"], None-safe
# ---------------------------------------------------------------------------
def test_extract_routed_experts_from_rollout_details():
    rd = [{"extra": {"routed_experts": [[_real_row(0)], [_real_row(1)]]}}]
    got = extract_routed_experts_from_rollout_details(rd)
    assert len(got) == 2
    np.testing.assert_array_equal(got[0], np.asarray([_real_row(0)], dtype=np.int16))
    np.testing.assert_array_equal(got[1], np.asarray([_real_row(1)], dtype=np.int16))

    # Missing / None cases all return None (treated as sentinel-filled sample).
    assert extract_routed_experts_from_rollout_details(None) is None
    assert extract_routed_experts_from_rollout_details([]) is None
    assert extract_routed_experts_from_rollout_details([{"logprobs": [[0.0]]}]) is None
    assert extract_routed_experts_from_rollout_details([{"extra": {}}]) is None
    assert extract_routed_experts_from_rollout_details([{"extra": {"routed_experts": []}}]) is None


# ---------------------------------------------------------------------------
# Case 3: packer — [batch, response_len, L, K], right-padded, shape[:2]==loss_mask
# ---------------------------------------------------------------------------
@pytest.fixture
def char_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    return tok


def test_packer_shape_and_right_pad(char_tokenizer):
    prompts = [[97, 98, 99], [49, 50, 51, 52, 53]]
    responses = [[100, 101, 102], [54, 55, 56, 57, 58]]
    rewards = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0, 0, 0, 0])]
    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]

    # Per-sample [resp_len_i, L, K] routed_experts.
    re0 = [_real_row(i) for i in range(3)]
    re1 = [_real_row(10 + i) for i in range(5)]
    routed_experts = [re0, re1]

    (
        sequences,
        attention_mask,
        action_mask,
        ret_rewards,
        ret_loss_masks,
        logprobs_tensor,
        routed_experts_tensor,
        _token_level_shaping_tensor,
        _response_span_tags_tensor,
    ) = convert_prompts_responses_to_batch_tensors(
        char_tokenizer, prompts, responses, rewards, loss_masks, None, routed_experts
    )

    max_resp = action_mask.size(1)
    # Invariant #1: shape == (batch, response_len, L, K).
    assert routed_experts_tensor.shape == (2, max_resp, L, K)
    # Invariant #2: routed_experts.shape[:2] == loss_masks.shape.
    assert tuple(routed_experts_tensor.shape[:2]) == tuple(ret_loss_masks.shape)
    # Width-minimized int dtype (R3 by-value spill fix): the tensor is narrowed to
    # the smallest int dtype that fits the max expert id present — uint8 when all
    # ids ≤ 255 (this fixture's ids are %7+1, so uint8), int16 up to 32767, else
    # int64. The training-side consumer upcasts back to int64, so this is a pure
    # transport-size optimization.
    assert routed_experts_tensor.dtype == torch.uint8
    # Round-trips exactly to the original int64 values on upcast (what the consumer does).
    assert torch.equal(
        routed_experts_tensor.to(torch.long),
        torch.tensor(routed_experts_tensor.tolist(), dtype=torch.long),
    )
    # Right-pad: sample 0 (len 3) padded to max_resp with sentinel rows.
    for t in range(3, max_resp):
        assert torch.all(routed_experts_tensor[0, t] == SENTINEL_EXPERT_ID)
    # Real rows preserved for the un-padded prefix.
    assert routed_experts_tensor[0, 0].tolist() == re0[0]
    assert routed_experts_tensor[1, 4].tolist() == re1[4]


# ---------------------------------------------------------------------------
# Case 4: no-op — routed_experts=None ⇒ collator output identical to today
# ---------------------------------------------------------------------------
def test_packer_noop_flag_off(char_tokenizer):
    prompts = [[97, 98, 99], [49, 50, 51, 52, 53]]
    responses = [[100, 101, 102], [54, 55, 56, 57, 58]]
    rewards = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0, 0, 0, 0])]
    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]

    # With routed_experts omitted (flag off), the 7th return is None and the first
    # six tensors are byte-identical to a pre-rail call.
    res_off = convert_prompts_responses_to_batch_tensors(char_tokenizer, prompts, responses, rewards, loss_masks)
    assert len(res_off) == 9
    assert res_off[6] is None  # routed_experts_tensor
    assert res_off[7] is None  # token_level_shaping_tensor (Stage B, off)
    assert res_off[8] is None  # response_span_tags_tensor (Stage B, off)

    res_off2 = convert_prompts_responses_to_batch_tensors(
        char_tokenizer, prompts, responses, rewards, loss_masks, None, None
    )
    for a, b in zip(res_off[:6], res_off2[:6]):
        if a is None and b is None:
            continue
        assert torch.equal(a, b)
    assert res_off2[6] is None


# ---------------------------------------------------------------------------
# Case 4b: TrainingInputBatch byte-identical when the flag is off (TensorBatch.__eq__).
# At the batch boundary, the flag-off batch must have the same keys and tensors,
# while the flag-on batch adds exactly one key (rollout_routed_experts).
# ---------------------------------------------------------------------------
def test_training_input_batch_noop_vs_present(char_tokenizer):
    from skyrl_train.training_batch import TrainingInputBatch

    prompts = [[97, 98, 99], [49, 50, 51, 52, 53]]
    responses = [[100, 101, 102], [54, 55, 56, 57, 58]]
    rewards = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0, 0, 0, 0])]
    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]
    re0 = [_real_row(i) for i in range(3)]
    re1 = [_real_row(10 + i) for i in range(5)]

    def build(routed_experts):
        # Mirror trainer.convert_to_training_input's tensor wiring (subset). Use
        # a real logprobs tensor so every present field is a Tensor (TensorBatch
        # .__eq__ does torch.equal over present keys).
        logprobs = [[0.0] * len(r) for r in responses]
        (seq, attn, resp_mask, rew, lm, lp, re_t, _tls, _rst) = convert_prompts_responses_to_batch_tensors(
            char_tokenizer, prompts, responses, rewards, loss_masks, logprobs, routed_experts
        )
        batch = TrainingInputBatch(
            {
                "sequences": seq,
                "attention_mask": attn,
                "response_mask": resp_mask,
                "rewards": rew,
                "loss_mask": lm,
                "rollout_logprobs": lp,
            }
        )
        if re_t is not None:
            batch["rollout_routed_experts"] = re_t
        return batch

    off_a = build(None)
    off_b = build(None)
    # Flag off twice → byte-identical (no extra key).
    assert off_a == off_b
    assert "rollout_routed_experts" not in off_a

    on = build([re0, re1])
    # Flag on adds exactly one key; otherwise NOT equal to the flag-off batch.
    assert "rollout_routed_experts" in on
    assert on != off_a
    # All shared keys remain byte-identical (purely additive).
    for k in off_a.keys():
        assert torch.equal(off_a[k], on[k])
    # The added field satisfies the Stage-1 invariant shape[:2] == loss_mask.shape.
    assert tuple(on["rollout_routed_experts"].shape[:2]) == tuple(on["loss_mask"].shape)


# ===========================================================================
# np.int16 array-carrier parity gates.
#
# These are the mandatory byte-identical gates from the fix spec §3(i)(ii): the
# collated rollout_routed_experts tensor (and the whole TrainingInputBatch) must be
# nested-list and array-carrier inputs remain bit-for-bit equal. Only the container
# type of the ids on the wire changes.
# ===========================================================================


def _sample_nested(n, seed0):
    """A per-sample [n, L, K] nested-list routed_experts block (real rows)."""
    return [_real_row(seed0 + i) for i in range(n)]


@pytest.mark.parametrize("num_experts", [None, 128, 512], ids=["none", "uint8", "int16"])
def test_collator_container_parity_byte_identical(char_tokenizer, num_experts):
    """Nested lists and np.int16 arrays produce a
    BIT-FOR-BIT identical rollout_routed_experts tensor + identical every-other
    returned tensor, across the deterministic-dtype variants (uint8 / int16) and the
    None per-batch-max fallback."""
    prompts = [[97, 98, 99], [49, 50, 51, 52, 53]]
    responses = [[100, 101, 102], [54, 55, 56, 57, 58]]
    rewards = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0, 0, 0, 0])]
    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]

    re_nested = [_sample_nested(3, 0), _sample_nested(5, 10)]
    # Flag-on carrier: per-sample contiguous np.int16 arrays (what the capture
    # boundary now emits). Same ids/rows, different container type only.
    re_arrays = [np.asarray(s, dtype=np.int16) for s in re_nested]

    off = convert_prompts_responses_to_batch_tensors(
        char_tokenizer, prompts, responses, rewards, loss_masks, None, re_nested, num_experts=num_experts
    )
    on = convert_prompts_responses_to_batch_tensors(
        char_tokenizer, prompts, responses, rewards, loss_masks, None, re_arrays, num_experts=num_experts
    )

    t_off, t_on = off[6], on[6]
    assert t_off is not None and t_on is not None
    assert t_off.dtype == t_on.dtype, f"dtype drift: {t_off.dtype} vs {t_on.dtype}"
    assert t_off.shape == t_on.shape == (2, 5, L, K)
    assert torch.equal(t_off, t_on), "array-carrier routed_experts must be byte-identical to nested input"
    for a, b in zip(off, on):
        if a is None and b is None:
            continue
        assert torch.equal(a, b)


def test_training_input_batch_container_parity(char_tokenizer):
    """Full TrainingInputBatch parity: same key set + every tensor torch.equal between
    the flag-off (nested) and flag-on (array) builds."""
    from skyrl_train.training_batch import TrainingInputBatch

    prompts = [[97, 98, 99], [49, 50, 51, 52, 53]]
    responses = [[100, 101, 102], [54, 55, 56, 57, 58]]
    rewards = [torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0, 0, 0, 0])]
    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]
    logprobs = [[0.0] * len(r) for r in responses]
    re_nested = [_sample_nested(3, 0), _sample_nested(5, 10)]
    re_arrays = [np.asarray(s, dtype=np.int16) for s in re_nested]

    def build(routed_experts, num_experts=512):
        (seq, attn, resp_mask, rew, lm, lp, re_t, _tls, _rst) = convert_prompts_responses_to_batch_tensors(
            char_tokenizer, prompts, responses, rewards, loss_masks, logprobs, routed_experts, num_experts=num_experts
        )
        batch = TrainingInputBatch(
            {
                "sequences": seq,
                "attention_mask": attn,
                "response_mask": resp_mask,
                "rewards": rew,
                "loss_mask": lm,
                "rollout_logprobs": lp,
            }
        )
        if re_t is not None:
            batch["rollout_routed_experts"] = re_t
        return batch

    off = build(re_nested)
    on = build(re_arrays)

    assert set(off.keys()) == set(on.keys())
    assert "rollout_routed_experts" in on
    for k in off.keys():
        assert torch.equal(off[k], on[k]), f"key {k} diverged between flag-off and flag-on"


def test_align_array_twin_parity():
    """align_routed_experts_with_lcs array twin: .tolist() bit-identical to the
    list branch across the LCS-mismatch, exact-1:1, and empty cases."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    retok_ids = [101, 102, 103, 104]
    vllm_rows = [_real_row(i) for i in range(5)]  # N+1 (count mismatch -> positional LCS)

    def _off(retok, rows):
        return align_routed_experts_with_lcs(retok, rows, tokenizer)

    def _on(retok, rows):
        return align_routed_experts_with_lcs(retok, np.asarray(rows, dtype=np.int16), tokenizer)

    # Count-mismatch positional LCS.
    off = _off(retok_ids, vllm_rows)
    on = _on(retok_ids, vllm_rows)
    assert isinstance(on, np.ndarray)
    assert on.dtype == np.int16
    assert np.asarray(off, dtype=np.int16).tolist() == on.tolist()

    # Exact 1:1 direct copy.
    off_eq = _off(retok_ids, vllm_rows[:4])
    on_eq = _on(retok_ids, vllm_rows[:4])
    assert np.asarray(off_eq, dtype=np.int16).tolist() == on_eq.tolist()

    # Empty vLLM rows -> [] both branches (caller sentinel-pads).
    assert align_routed_experts_with_lcs(retok_ids, [], tokenizer) == []
    assert align_routed_experts_with_lcs(retok_ids, np.zeros((0, L, K), dtype=np.int16), tokenizer) == []


def test_end_to_end_alignment_and_collate_container_parity(char_tokenizer):
    """END-TO-END gate: multi-turn get_response_ids_and_loss_mask_from_messages +
    collator, flag-off (nested per-turn) vs flag-on (np.int16 per-turn arrays), must
    yield a byte-identical collated routed_experts tensor. Exercises extract-shape,
    sentinel-fill, LCS alignment, AND the collator — the correctness-sensitive path."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    generation_prompt_ids = get_generation_prompt_ids(tokenizer)

    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Good"},
    ]

    def num_generated(content):
        msg = [{"role": "assistant", "content": content}]
        ids = encode_messages_subset(msg, tokenizer)
        last_eos = len(ids) - 1 - ids[::-1].index(tokenizer.eos_token_id)
        return last_eos + 1 - len(generation_prompt_ids)

    n1 = num_generated("Hello there")
    n2 = num_generated("Good")
    re_turn1 = [_real_row(10 + i) for i in range(n1)]
    re_turn2 = [_real_row(50 + i) for i in range(n2)]

    def assemble(per_turn):
        _, _, _, routed = get_response_ids_and_loss_mask_from_messages(
            messages, tokenizer, assistant_routed_experts=per_turn
        )
        return routed

    # Nested and array inputs are both normalized to the canonical array carrier.
    routed_off = assemble([re_turn1, re_turn2])
    # Per-turn np.int16 arrays are what the capture boundary emits.
    routed_on = assemble([np.asarray(re_turn1, dtype=np.int16), np.asarray(re_turn2, dtype=np.int16)])

    # Collate each sample-of-one.
    prompts = [[1, 2, 3]]
    responses = [list(range(100, 100 + len(routed_off)))]
    rewards = [torch.zeros(len(routed_off))]
    loss_masks = [[1] * len(routed_off)]

    t_off = convert_prompts_responses_to_batch_tensors(
        char_tokenizer, prompts, responses, rewards, loss_masks, None, [routed_off], num_experts=512
    )[6]
    t_on = convert_prompts_responses_to_batch_tensors(
        char_tokenizer, prompts, responses, rewards, loss_masks, None, [routed_on], num_experts=512
    )[6]

    assert t_off is not None and t_on is not None
    assert t_off.shape == t_on.shape
    assert t_off.dtype == t_on.dtype
    assert torch.equal(t_off, t_on), "array-carrier tensor must be byte-identical to nested input"


def test_concat_cross_sample_sentinel_matches_LK():
    """Regression for #232: concatenate_trajectory_batches must sentinel-fill a
    routing-less generator output with [L, K]-shaped rows learned from a sibling
    output that DOES carry routing, not a degenerate [1, 1] row. A [1, 1] sentinel
    mixed with real [L, K] rows makes the L axis ragged and crashes the downstream
    dense torch.tensor() collation ("expected sequence of length 1 at dim 2").
    """
    real_row = [[1] * K for _ in range(L)]  # [L, K]
    # Output A carries routing; output B does not (preempted/quant path).
    out_a = {
        "prompt_token_ids": [[10, 11]],
        "response_ids": [[20, 21, 22]],
        "rewards": [[0.0, 0.0, 1.0]],
        "loss_masks": [[1, 1, 1]],
        "rollout_routed_experts": [[real_row, real_row, real_row]],
    }
    out_b = {
        "prompt_token_ids": [[30, 31]],
        "response_ids": [[40, 41]],
        "rewards": [[0.0, 1.0]],
        "loss_masks": [[1, 1]],
        # no rollout_routed_experts key
    }

    merged = concatenate_trajectory_batches([out_a, out_b])
    assert "rollout_routed_experts" in merged
    re = merged["rollout_routed_experts"]
    assert len(re) == 2  # one entry per sample
    # The sentinel-filled sample (B) must have [L, K]-shaped rows, NOT [1, 1].
    sentinel_sample = re[1]
    assert len(sentinel_sample) == 2  # 1:1 with response_ids
    for row in sentinel_sample:
        assert len(row) == L
        assert all(len(layer) == K for layer in row)
        assert all(all(e == SENTINEL_EXPERT_ID for e in layer) for layer in row)
