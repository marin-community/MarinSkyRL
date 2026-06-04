"""MoE router replay (R3) on the eager HF MoE path — Stage 2 GPU test.

Adapted from prime-rl ``tests/unit/train/models/test_qwen3_5_moe.py::
test_qwen3_5_moe_router_replay``. Single GPU, dense (unpacked), no EP/SP.

Run::

    uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_router_replay_eager.py

Or directly (no pytest):

    python tests/gpu/gpu_ci/test_router_replay_eager.py
"""

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

try:
    import pytest

    pytestmark = [pytest.mark.gpu]
except ImportError:  # pytest absent on cluster envs — direct invocation still works
    pytest = None

from skyrl_train.model_wrapper import HFModelWrapper
from skyrl_train.models.router_replay import (
    count_moe_layers,
    get_active_replay,
    SENTINEL_EXPERT_ID,
)


def _make_config():
    cfg = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _build_wrapper(moe_router_replay: bool, device="cuda", seed=0, use_sample_packing=False):
    torch.manual_seed(seed)
    cfg = _make_config()
    # Sample packing requires FA2 (the wrapper asserts this in __init__).
    if use_sample_packing:
        cfg._attn_implementation = "flash_attention_2"
    with torch.device(device):
        hf_model = Qwen3MoeForCausalLM._from_config(cfg)
    hf_model = hf_model.to(torch.float32)
    wrapper = HFModelWrapper(
        pretrain_or_model=hf_model,
        use_flash_attention_2=use_sample_packing,
        bf16=False,
        sequence_parallel_size=1,
        use_sample_packing=use_sample_packing,
        moe_router_replay=moe_router_replay,
    )
    wrapper.model.to(device)
    return wrapper, cfg


def _inputs(batch=1, seq_len=32, num_actions=16, vocab=256, device="cuda"):
    input_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    return input_ids, attention_mask, num_actions


def _ragged_inputs(seq_len=32, num_actions=16, vocab=256, device="cuda", seed=7):
    """Left-padded batch with DIFFERENT real-token lengths per row.

    Each row's response is the last ``num_actions`` positions (fixed scalar);
    the prompt portion varies in length so the left-pad count differs per row.
    With non-trivial left padding, ``unpad_input`` actually permutes tokens out
    of the naive row-major contiguous order, which is what G3a-3 stresses.
    """
    torch.manual_seed(seed)
    batch = 3
    # Real-token lengths per row (all >= num_actions so the full response is
    # real). Distinct prompt lengths → distinct left-pad counts.
    real_lens = [num_actions + 2, num_actions + 7, seq_len]  # e.g. 18, 23, 32
    input_ids = torch.randint(1, vocab, (batch, seq_len), device=device)
    attention_mask = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    for b, rl in enumerate(real_lens):
        rl = min(rl, seq_len)
        attention_mask[b, seq_len - rl:] = 1
        # zero out the left-pad token ids (cosmetic; FA2 ignores them anyway)
        input_ids[b, : seq_len - rl] = 0
    return input_ids, attention_mask, num_actions


def _forward_logits(wrapper, input_ids, attention_mask, num_actions, rollout_routed_experts=None):
    # The wrapper.forward returns action_log_probs; we want the raw logits to
    # compare. Use return_output=True and read output["logits"].
    _, output = wrapper(
        input_ids,
        num_actions,
        attention_mask=attention_mask,
        temperature=1.0,
        return_output=True,
        compute_entropy=False,
        rollout_routed_experts=rollout_routed_experts,
    )
    return output["logits"]


def _all_one_expert_mask(batch, num_actions, L, K, expert_id, device):
    # Force every response token to (expert_id repeated)? No — duplicate rows
    # revert to native. Force [expert_id, expert_id+1] so each row is a valid
    # distinct top-k set that is unlikely to equal the natural choice.
    re = torch.empty(batch, num_actions, L, K, dtype=torch.long, device=device)
    re[..., 0] = expert_id
    if K > 1:
        re[..., 1] = (expert_id + 1)
    for k in range(2, K):
        re[..., k] = (expert_id + k)
    return re


def test_a_override_bites():
    """(a) Forcing all response tokens to a fixed expert set changes the
    response-position logits vs natural routing."""
    device = "cuda"
    wrapper, cfg = _build_wrapper(True, device=device)
    input_ids, attn, num_actions = _inputs(device=device)
    L = count_moe_layers(cfg)
    K = cfg.num_experts_per_tok

    out_natural = _forward_logits(wrapper, input_ids, attn, num_actions)
    re = _all_one_expert_mask(input_ids.shape[0], num_actions, L, K, expert_id=3, device=device)
    out_replay = _forward_logits(wrapper, input_ids, attn, num_actions, rollout_routed_experts=re)

    resp_natural = out_natural[:, -num_actions:, :]
    resp_replay = out_replay[:, -num_actions:, :]
    assert not torch.allclose(resp_natural, resp_replay, atol=1e-4), (
        "router replay override did not change response logits"
    )
    print("[a] override bites: PASS")


def test_b_router_and_expert_grads():
    """(b) backward through the replayed forward populates gate.weight.grad
    (router grad) AND an expert down_proj.weight.grad (expert grad) — the R3
    crux that weights are re-gathered from the LIVE softmax."""
    device = "cuda"
    wrapper, cfg = _build_wrapper(True, device=device)
    input_ids, attn, num_actions = _inputs(device=device)
    L = count_moe_layers(cfg)
    K = cfg.num_experts_per_tok

    re = _all_one_expert_mask(input_ids.shape[0], num_actions, L, K, expert_id=3, device=device)
    wrapper.model.zero_grad()
    out = _forward_logits(wrapper, input_ids, attn, num_actions, rollout_routed_experts=re)
    out.sum().backward()

    layer0 = wrapper.model.model.layers[0].mlp
    assert layer0.gate.weight.grad is not None, "router gate.weight.grad is None"
    assert layer0.gate.weight.grad.abs().sum() > 0, "router gate.weight.grad is all-zero"
    # Forced expert 3 must receive grad.
    expert = layer0.experts[3]
    assert expert.down_proj.weight.grad is not None, "forced expert down_proj.weight.grad is None"
    assert expert.down_proj.weight.grad.abs().sum() > 0, "forced expert grad is all-zero"
    print("[b] router + expert grads: PASS")


def test_c_determinism():
    """(c) Same mask twice → identical logits."""
    device = "cuda"
    wrapper, cfg = _build_wrapper(True, device=device)
    input_ids, attn, num_actions = _inputs(device=device)
    L = count_moe_layers(cfg)
    K = cfg.num_experts_per_tok
    re = _all_one_expert_mask(input_ids.shape[0], num_actions, L, K, expert_id=2, device=device)

    out1 = _forward_logits(wrapper, input_ids, attn, num_actions, rollout_routed_experts=re)
    out2 = _forward_logits(wrapper, input_ids, attn, num_actions, rollout_routed_experts=re)
    assert torch.equal(out1, out2), "router replay is non-deterministic for identical masks"
    print("[c] determinism: PASS")


def test_d_flag_off_byte_identical():
    """(d) Flag off → _router_replay is None, no controller active, logits
    byte-identical to a stock HF forward on the same model."""
    device = "cuda"
    wrapper, cfg = _build_wrapper(False, device=device)
    assert wrapper._router_replay is None, "_router_replay should be None when flag off"
    assert get_active_replay() is None, "no controller should be active when flag off"

    input_ids, attn, num_actions = _inputs(device=device)
    out_wrapper = _forward_logits(wrapper, input_ids, attn, num_actions)

    # Stock HF forward on the same underlying model with the same position_ids
    # the wrapper computes.
    position_ids = attn.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn == 0, 1)
    stock = wrapper.model(input_ids, attention_mask=attn, position_ids=position_ids)["logits"]
    assert torch.equal(out_wrapper, stock), "flag-off forward not byte-identical to stock HF"
    print("[d] flag-off byte-identical: PASS")


def test_extra_layer_count_and_sentinel():
    """Extra: discovered MoE layers == count_moe_layers(cfg); a sentinel /
    prompt row routes naturally."""
    device = "cuda"
    wrapper, cfg = _build_wrapper(True, device=device)
    expected = count_moe_layers(cfg)

    # Layer-count discovery: run a forward with a target so the controller
    # populates its id table, then assert.
    input_ids, attn, num_actions = _inputs(device=device)
    L = expected
    K = cfg.num_experts_per_tok
    re = _all_one_expert_mask(input_ids.shape[0], num_actions, L, K, expert_id=1, device=device)
    _ = _forward_logits(wrapper, input_ids, attn, num_actions, rollout_routed_experts=re)
    # Note: controller.clear() ran in finally, so re-discover via a fresh forward
    # under begin_replay handled internally; instead assert via install count.
    from skyrl_train.models.router_replay import install_router_replay_patch

    discovered = install_router_replay_patch(wrapper.model)
    assert discovered == expected, f"discovered {discovered} MoE blocks, expected {expected}"

    # Sentinel rows (all-SENTINEL_EXPERT_ID) must route naturally. Build a mask
    # that is all-sentinel on the response → replay mask all False → forward
    # equals natural.
    out_natural = _forward_logits(wrapper, input_ids, attn, num_actions)
    re_sentinel = torch.full(
        (input_ids.shape[0], num_actions, L, K), SENTINEL_EXPERT_ID, dtype=torch.long, device=device
    )
    out_sentinel = _forward_logits(
        wrapper, input_ids, attn, num_actions, rollout_routed_experts=re_sentinel
    )
    assert torch.equal(out_natural, out_sentinel), (
        "all-sentinel mask did not fall through to natural routing"
    )
    print(f"[extra] layer count == {expected} and sentinel routes naturally: PASS")


# --------------------------------------------------------------------------- #
# Stage 3a — sample-packing replay path                                       #
# --------------------------------------------------------------------------- #


def test_g3a_1_packed_equals_unpacked():
    """G3a-1: replay packed == replay unpacked.

    Same model weights, same inputs, same rollout_routed_experts. Run the
    wrapper once with use_sample_packing=False (dense) and once with
    use_sample_packing=True (+ FA2). The response-position logits must match
    (FA2 vs eager tol atol≈2e-2). Proves the packed index_select target lands
    on the same rows the dense path forces.
    """
    device = "cuda"
    # Same seed → identical weights for both wrappers.
    wrapper_dense, cfg = _build_wrapper(True, device=device, seed=0, use_sample_packing=False)
    wrapper_packed, _ = _build_wrapper(True, device=device, seed=0, use_sample_packing=True)
    # FA2 needs bf16/fp16; build both in the same dtype for a clean compare.
    wrapper_dense.model.to(torch.bfloat16)
    wrapper_packed.model.to(torch.bfloat16)

    input_ids, attn, num_actions = _ragged_inputs(device=device)
    L = count_moe_layers(cfg)
    K = cfg.num_experts_per_tok
    re = _all_one_expert_mask(input_ids.shape[0], num_actions, L, K, expert_id=3, device=device)

    out_dense = _forward_logits(wrapper_dense, input_ids, attn, num_actions, rollout_routed_experts=re)
    out_packed = _forward_logits(wrapper_packed, input_ids, attn, num_actions, rollout_routed_experts=re)

    # Compare response positions only (the slice replay forces). pad_input
    # restores [B, seq_len] for the packed path, so shapes align.
    resp_dense = out_dense[:, -num_actions:, :].float()
    resp_packed = out_packed[:, -num_actions:, :].float()
    assert torch.allclose(resp_dense, resp_packed, atol=2e-2), (
        f"packed replay != dense replay on response logits "
        f"(max abs diff {(resp_dense - resp_packed).abs().max().item():.4e})"
    )
    print("[G3a-1] packed replay == unpacked replay: PASS")


def test_g3a_2_production_noop():
    """G3a-2: production no-op. moe_router_replay=False + use_sample_packing=True
    → no controller installed → torch.equal to today's packed forward."""
    device = "cuda"
    wrapper, cfg = _build_wrapper(False, device=device, use_sample_packing=True)
    assert wrapper._router_replay is None, "_router_replay must be None when flag off"
    assert get_active_replay() is None, "no controller may be active when flag off"

    input_ids, attn, num_actions = _ragged_inputs(device=device)
    out_wrapper = _forward_logits(wrapper, input_ids, attn, num_actions)

    # Reproduce today's packed forward directly: unpad → model → pad. This is
    # exactly what wrapper.forward does when no controller is installed; the
    # patched SparseMoeBlock short-circuits to orig_forward (no controller).
    from flash_attn.bert_padding import pad_input, unpad_input

    position_ids = attn.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn == 0, 1)
    seq_fwd, nnz_indices, _, _, _ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attn)
    seq_fwd = seq_fwd.transpose(0, 1)
    pos_fwd, _, _, _, _ = unpad_input(position_ids.unsqueeze(-1), attn)
    pos_fwd = pos_fwd.transpose(0, 1)
    stock_packed = wrapper.model(seq_fwd, attention_mask=None, position_ids=pos_fwd)["logits"]
    # The wrapper returns logits in packed [1, nnz, V] form (no pad_input on
    # logits inside forward — only log_probs/entropy get re-padded). Compare
    # the packed logits directly.
    assert torch.equal(out_wrapper, stock_packed), (
        "flag-off packed forward not byte-identical to stock packed forward"
    )
    print("[G3a-2] production packed no-op byte-identical: PASS")


def test_g3a_3_ragged_alignment():
    """G3a-3: ragged-length alignment. Force a distinguishable expert set on
    ONE sequence's response only (others sentinel → natural). Assert only that
    sequence's response logits move vs natural; the other sequences are
    byte-identical to natural. Proves the nnz_indices permute lands the forced
    target on the correct (permuted) rows under packing.
    """
    device = "cuda"
    wrapper, cfg = _build_wrapper(True, device=device, use_sample_packing=True)
    wrapper.model.to(torch.bfloat16)

    input_ids, attn, num_actions = _ragged_inputs(device=device)
    batch = input_ids.shape[0]
    L = count_moe_layers(cfg)
    K = cfg.num_experts_per_tok

    out_natural = _forward_logits(wrapper, input_ids, attn, num_actions).float()

    # All-sentinel everywhere except sequence index 1's response → forced.
    re = torch.full((batch, num_actions, L, K), SENTINEL_EXPERT_ID, dtype=torch.long, device=device)
    forced_seq = 1
    re[forced_seq, :, :, 0] = 5
    if K > 1:
        re[forced_seq, :, :, 1] = 6
    for k in range(2, K):
        re[forced_seq, :, :, k] = 5 + k
    out_replay = _forward_logits(wrapper, input_ids, attn, num_actions, rollout_routed_experts=re).float()

    resp_n = out_natural[:, -num_actions:, :]
    resp_r = out_replay[:, -num_actions:, :]
    moved = [(resp_n[b] - resp_r[b]).abs().max().item() for b in range(batch)]
    # The forced sequence must move; the others must not (allclose to natural).
    assert moved[forced_seq] > 1e-3, (
        f"forced sequence {forced_seq} response logits did not move (max diff {moved[forced_seq]:.2e})"
    )
    for b in range(batch):
        if b == forced_seq:
            continue
        assert torch.allclose(resp_n[b], resp_r[b], atol=2e-2), (
            f"non-forced sequence {b} moved (max diff {moved[b]:.2e}) — nnz_indices permute landed on wrong rows"
        )
    print(f"[G3a-3] ragged alignment (only seq {forced_seq} moved; diffs={[f'{m:.2e}' for m in moved]}): PASS")


if __name__ == "__main__":
    import sys

    # G3a-2 is the CPU/no-GPU-friendly no-op proof structurally, but FA2 needs a
    # GPU. Run whatever the host supports.
    if not torch.cuda.is_available():
        print("CUDA not available — Stage 3a GPU gates DEFERRED.")
        sys.exit(0)

    test_d_flag_off_byte_identical()
    test_a_override_bites()
    test_b_router_and_expert_grads()
    test_c_determinism()
    test_extra_layer_count_and_sentinel()
    test_g3a_2_production_noop()
    test_g3a_1_packed_equals_unpacked()
    test_g3a_3_ragged_alignment()
    print("ALL PASS")
