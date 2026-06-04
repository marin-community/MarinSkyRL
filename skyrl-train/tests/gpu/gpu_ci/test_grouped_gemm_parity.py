"""Stage 3b — grouped-GEMM MoE swap parity gates (1-2 GPU, EP=1).

Gates (scope §3b):
  G3b-1  grouped MoE forward allclose to HF eager on identical weights/inputs
         (atol≈2e-2) + a backward step's grad diff bounded.
  G3b-2  replay-on grouped: override bites + router/expert grads + determinism,
         via the NATIVE router routed_experts arg (Stage-2 RouterReplay singleton
         transport).
  G3b-4  moe_grouped_gemm=False → unswapped → torch.equal to today's HF eager.

Run on a Qwen3-MoE tiny config AND a Qwen3-Next tiny config (the latter
exercises the shared-expert sigmoid-gate adaptation).

Run::

    uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_grouped_gemm_parity.py
    # or directly (no pytest): python tests/gpu/gpu_ci/test_grouped_gemm_parity.py
"""

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

try:
    import pytest

    pytestmark = [pytest.mark.gpu]
except ImportError:  # pytest absent on cluster envs — direct invocation still works
    pytest = None

from skyrl_train.model_wrapper import HFModelWrapper
from skyrl_train.models.router_replay import count_moe_layers, get_active_replay


# --------------------------------------------------------------------------- #
# Config builders                                                             #
# --------------------------------------------------------------------------- #


def _qwen3_moe_config():
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


def _build_qwen3_moe(device, seed):
    torch.manual_seed(seed)
    cfg = _qwen3_moe_config()
    with torch.device(device):
        model = Qwen3MoeForCausalLM._from_config(cfg).to(torch.float32)
    return model, cfg


def _build_qwen3_next(device, seed):
    """Tiny Qwen3-Next (shared-expert + sigmoid gate). Returns (model, cfg) or
    (None, None) if transformers lacks Qwen3-Next."""
    try:
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    except Exception:
        return None, None
    torch.manual_seed(seed)
    cfg = Qwen3NextConfig(
        vocab_size=256,
        hidden_size=256,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
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
    with torch.device(device):
        model = Qwen3NextForCausalLM._from_config(cfg).to(torch.float32)
    return model, cfg


def _wrap(model, *, moe_grouped_gemm, moe_router_replay, device):
    w = HFModelWrapper(
        pretrain_or_model=model,
        use_flash_attention_2=False,
        bf16=False,
        sequence_parallel_size=1,
        use_sample_packing=False,
        moe_router_replay=moe_router_replay,
        moe_grouped_gemm=moe_grouped_gemm,
    )
    w.model.to(device=device, dtype=torch.float32)
    return w


def _inputs(device, batch=1, seq_len=32, num_actions=16, vocab=256):
    input_ids = torch.randint(0, vocab, (batch, seq_len), device=device)
    attn = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    return input_ids, attn, num_actions


def _forward_logits(wrapper, input_ids, attn, num_actions, rollout_routed_experts=None):
    _, output = wrapper(
        input_ids, num_actions, attention_mask=attn, temperature=1.0,
        return_output=True, compute_entropy=False, rollout_routed_experts=rollout_routed_experts,
    )
    return output["logits"]


def _force_mask(batch, num_actions, L, K, expert_id, device):
    re = torch.empty(batch, num_actions, L, K, dtype=torch.long, device=device)
    re[..., 0] = expert_id
    for k in range(1, K):
        re[..., k] = expert_id + k
    return re


# --------------------------------------------------------------------------- #
# G3b-4 — flag-off no-op (the hard requirement)                               #
# --------------------------------------------------------------------------- #


def _g3b_4(build, name, device):
    model, cfg = build(device, seed=0)
    if model is None:
        print(f"[G3b-4/{name}] config unavailable — SKIP")
        return
    # Stock HF logits (reference) BEFORE any wrapper construction.
    input_ids, attn, num_actions = _inputs(device)
    position_ids = attn.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn == 0, 1)
    stock = model(input_ids, attention_mask=attn, position_ids=position_ids)["logits"].clone()

    w = _wrap(model, moe_grouped_gemm=False, moe_router_replay=False, device=device)
    # No swap: the .mlp blocks must still be HF *SparseMoeBlock, not the shim.
    for layer in w.model.model.layers:
        assert type(layer.mlp).__name__.endswith("SparseMoeBlock"), (
            f"[{name}] flag-off swapped a block (got {type(layer.mlp).__name__})"
        )
    out = _forward_logits(w, input_ids, attn, num_actions)
    assert torch.equal(out, stock), f"[{name}] flag-off forward not byte-identical to stock HF"
    print(f"[G3b-4/{name}] flag-off byte-identical (no swap): PASS")


def test_g3b_4_flag_off_qwen3_moe():
    _g3b_4(_build_qwen3_moe, "Qwen3-MoE", "cuda")


def test_g3b_4_flag_off_qwen3_next():
    _g3b_4(_build_qwen3_next, "Qwen3-Next", "cuda")


# --------------------------------------------------------------------------- #
# G3b-1 — grouped == HF eager (fwd + bwd)                                      #
# --------------------------------------------------------------------------- #


def _g3b_1(build, name, device):
    # Two identical models (same seed): one stays HF eager, one is swapped.
    eager_model, cfg = build(device, seed=0)
    if eager_model is None:
        print(f"[G3b-1/{name}] config unavailable — SKIP")
        return
    grouped_model, _ = build(device, seed=0)

    input_ids, attn, num_actions = _inputs(device)

    w_eager = _wrap(eager_model, moe_grouped_gemm=False, moe_router_replay=False, device=device)
    w_grouped = _wrap(grouped_model, moe_grouped_gemm=True, moe_router_replay=False, device=device)
    # The swap happened: blocks are shims.
    for layer in w_grouped.model.model.layers:
        assert type(layer.mlp).__name__ == "GroupedMoEShim", (
            f"[{name}] grouped wrapper did not swap (got {type(layer.mlp).__name__})"
        )

    out_eager = _forward_logits(w_eager, input_ids, attn, num_actions)
    out_grouped = _forward_logits(w_grouped, input_ids, attn, num_actions)
    diff = (out_eager - out_grouped).abs().max().item()
    assert torch.allclose(out_eager, out_grouped, atol=2e-2), (
        f"[{name}] grouped fwd != HF eager (max diff {diff:.4e})"
    )

    # Backward: grad on embed_tokens should be bounded-close.
    w_eager.model.zero_grad()
    w_grouped.model.zero_grad()
    _forward_logits(w_eager, input_ids, attn, num_actions).sum().backward()
    _forward_logits(w_grouped, input_ids, attn, num_actions).sum().backward()
    g_eager = w_eager.model.model.embed_tokens.weight.grad
    g_grouped = w_grouped.model.model.embed_tokens.weight.grad
    gdiff = (g_eager - g_grouped).abs().max().item()
    assert torch.allclose(g_eager, g_grouped, atol=2.0), f"[{name}] grad diff too large (max {gdiff:.4e})"
    print(f"[G3b-1/{name}] grouped == HF eager fwd (diff {diff:.2e}) + bwd (gdiff {gdiff:.2e}): PASS")


def test_g3b_1_qwen3_moe():
    _g3b_1(_build_qwen3_moe, "Qwen3-MoE", "cuda")


def test_g3b_1_qwen3_next():
    _g3b_1(_build_qwen3_next, "Qwen3-Next", "cuda")


# --------------------------------------------------------------------------- #
# G3b-2 — replay-on grouped (native router transport)                         #
# --------------------------------------------------------------------------- #


def _g3b_2(build, name, device):
    model, cfg = build(device, seed=0)
    if model is None:
        print(f"[G3b-2/{name}] config unavailable — SKIP")
        return
    w = _wrap(model, moe_grouped_gemm=True, moe_router_replay=True, device=device)
    assert w._router_replay is not None, f"[{name}] controller not created on grouped+replay path"

    input_ids, attn, num_actions = _inputs(device)
    L = count_moe_layers(cfg)
    K = cfg.num_experts_per_tok

    # (a) override bites
    out_natural = _forward_logits(w, input_ids, attn, num_actions)
    re = _force_mask(input_ids.shape[0], num_actions, L, K, expert_id=3, device=device)
    out_replay = _forward_logits(w, input_ids, attn, num_actions, rollout_routed_experts=re)
    rn = out_natural[:, -num_actions:, :]
    rr = out_replay[:, -num_actions:, :]
    assert not torch.allclose(rn, rr, atol=1e-4), f"[{name}] replay override did not change logits"

    # (b) router + expert grads (R3 crux: live-softmax re-gather)
    w.model.zero_grad()
    out = _forward_logits(w, input_ids, attn, num_actions, rollout_routed_experts=re)
    out.sum().backward()
    moe = w.model.model.layers[0].mlp.moe
    assert moe.router.gate.weight.grad is not None and moe.router.gate.weight.grad.abs().sum() > 0, (
        f"[{name}] router gate grad missing/zero"
    )
    assert moe.experts.w2.grad is not None and moe.experts.w2.grad.abs().sum() > 0, (
        f"[{name}] expert (w2/down) grad missing/zero"
    )

    # (c) determinism
    o1 = _forward_logits(w, input_ids, attn, num_actions, rollout_routed_experts=re)
    o2 = _forward_logits(w, input_ids, attn, num_actions, rollout_routed_experts=re)
    assert torch.equal(o1, o2), f"[{name}] grouped replay non-deterministic"
    assert get_active_replay() is None, f"[{name}] controller left active after forward"
    print(f"[G3b-2/{name}] replay override+grads+determinism (native router): PASS")


def test_g3b_2_qwen3_moe():
    _g3b_2(_build_qwen3_moe, "Qwen3-MoE", "cuda")


def test_g3b_2_qwen3_next():
    _g3b_2(_build_qwen3_next, "Qwen3-Next", "cuda")


if __name__ == "__main__":
    import sys

    if not torch.cuda.is_available():
        print("CUDA not available — Stage 3b GPU gates (G3b-1/2/4) DEFERRED.")
        sys.exit(0)

    for build, name in ((_build_qwen3_moe, "Qwen3-MoE"), (_build_qwen3_next, "Qwen3-Next")):
        _g3b_4(build, name, "cuda")
        _g3b_1(build, name, "cuda")
        _g3b_2(build, name, "cuda")
    print("ALL PASS")
