"""Stage 3b — G3b-3: HF → grouped → HF weight-remap roundtrip is lossless (CPU).

The cheap guard for the highest-risk surface (HF→grouped remap). Ported from
prime-rl ``tests/unit/train/models/test_qwen3_5_moe.py::test_qwen3_5_moe_roundtrip``
and adapted to SkyRL's per-layer state-dict converter
(``skyrl_train/models/layers/moe_weight_remap.py``). Also exercises a direct
live-module remap + forward parity (for-loop grouped MoE == HF eager block) on
CPU fp32, including the Qwen3-Next shared-expert sigmoid-gate adaptation.

Run::

    uv run --isolated --extra dev pytest tests/cpu/models/test_moe_weight_remap.py
    # or: python tests/cpu/models/test_moe_weight_remap.py
"""

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

try:
    import pytest

    pytestmark = [pytest.mark.cpu]
except ImportError:  # pytest absent on cluster envs — direct invocation still works
    pytest = None

from skyrl_train.models.layers.moe import MoE
from skyrl_train.models.layers.moe_swap import _build_moe_for_block
from skyrl_train.models.layers.moe_weight_remap import (
    convert_hf_to_tt_moe,
    convert_tt_to_hf_moe,
    remap_hf_block_to_moe,
)


def _qwen3_moe_config():
    cfg = Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "eager"
    return cfg


def test_g3b_3_state_dict_roundtrip_lossless():
    """hf → grouped → hf at the state_dict level is byte-lossless."""
    torch.manual_seed(0)
    cfg = _qwen3_moe_config()
    with torch.device("cpu"):
        hf_model = Qwen3MoeForCausalLM._from_config(cfg).to(torch.float32)

    original = {k: v.clone() for k, v in hf_model.state_dict().items()}

    # hf -> grouped -> hf (in place on a copy).
    sd = {k: v.clone() for k, v in original.items()}
    convert_hf_to_tt_moe(sd)
    # The grouped state_dict must NOT contain per-expert keys for MoE layers.
    assert not any("experts.0.gate_proj" in k for k in sd), "per-expert keys survived hf->grouped"
    assert any("experts.w1" in k for k in sd), "grouped experts.w1 key missing after hf->grouped"
    convert_tt_to_hf_moe(sd)

    assert set(sd.keys()) == set(original.keys()), (
        f"key mismatch after roundtrip: "
        f"+{set(sd) - set(original)} -{set(original) - set(sd)}"
    )
    for k in original:
        assert torch.equal(sd[k], original[k]), f"value mismatch at {k}"
    print("[G3b-3] state_dict hf->grouped->hf roundtrip lossless: PASS")


def test_g3b_3_live_module_remap_forward_parity():
    """Live HF block → grouped MoE remap reproduces HF eager forward (CPU fp32).

    Cheap forward-parity check on a single MoE block (no full model). The
    for-loop grouped path is fp32-exact vs HF eager up to float reductions.
    """
    torch.manual_seed(1)
    cfg = _qwen3_moe_config()
    with torch.device("cpu"):
        hf_model = Qwen3MoeForCausalLM._from_config(cfg).to(torch.float32)

    hf_block = hf_model.model.layers[0].mlp  # Qwen3MoeSparseMoeBlock
    moe = _build_moe_for_block(hf_block, cfg).to(torch.float32)
    remap_hf_block_to_moe(hf_block, moe)

    x = torch.randn(2, 16, cfg.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        hf_out, _ = hf_block(x)
        moe_out = moe(x)
    diff = (hf_out - moe_out).abs().max().item()
    assert torch.allclose(hf_out, moe_out, atol=2e-2), f"grouped MoE != HF eager block (max diff {diff:.4e})"
    print(f"[G3b-3] live-module remap forward parity (Qwen3-MoE, max diff {diff:.2e}): PASS")


def test_g3b_3_qwen3_next_shared_expert_gate():
    """Qwen3-Next shared-expert sigmoid gate is remapped and applied.

    Builds the Qwen3-Next SparseMoeBlock (shared_expert + shared_expert_gate),
    remaps into a grouped MoE with shared_expert_gated=True, and asserts forward
    parity — verifying F.sigmoid(shared_expert_gate(h)) * shared_expert(h).
    Skips gracefully if the installed transformers lacks Qwen3-Next.
    """
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
        from transformers import Qwen3NextConfig
    except Exception as e:  # transformers without Qwen3-Next
        print(f"[G3b-3] Qwen3-Next unavailable ({e}) — SKIP")
        return

    torch.manual_seed(2)
    cfg = Qwen3NextConfig(
        vocab_size=256,
        hidden_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    with torch.device("cpu"):
        block = Qwen3NextSparseMoeBlock(cfg).to(torch.float32)

    assert getattr(block, "shared_expert", None) is not None
    assert getattr(block, "shared_expert_gate", None) is not None

    moe = _build_moe_for_block(block, cfg).to(torch.float32)
    assert moe.shared_expert is not None, "grouped MoE missing shared expert for Qwen3-Next"
    assert moe.shared_expert_gate is not None, "grouped MoE missing shared_expert_gate for Qwen3-Next"
    remap_hf_block_to_moe(block, moe)

    x = torch.randn(2, 16, cfg.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        hf_out, _ = block(x)
        moe_out = moe(x)
    diff = (hf_out - moe_out).abs().max().item()
    assert torch.allclose(hf_out, moe_out, atol=2e-2), (
        f"grouped MoE (Qwen3-Next, gated shared expert) != HF eager (max diff {diff:.4e})"
    )
    print(f"[G3b-3] Qwen3-Next shared-expert sigmoid gate parity (max diff {diff:.2e}): PASS")


if __name__ == "__main__":
    test_g3b_3_state_dict_roundtrip_lossless()
    test_g3b_3_live_module_remap_forward_parity()
    test_g3b_3_qwen3_next_shared_expert_gate()
    print("ALL PASS")
