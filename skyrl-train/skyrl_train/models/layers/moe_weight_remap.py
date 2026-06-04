"""HF ``*SparseMoeBlock`` → grouped-GEMM ``MoE`` weight remap — Stage 3b.

Ported from prime-rl ``converting_qwen3_5_moe.py`` (the
``convert_hf_layer_to_tt`` / ``convert_tt_layer_to_hf`` per-layer logic).

Two surfaces:

  * ``remap_hf_block_to_moe(hf_block, moe)`` — copy a live HF
    ``Qwen3MoeSparseMoeBlock`` / ``Qwen3NextSparseMoeBlock`` instance's weights
    into a freshly-built grouped ``MoE``. Used by the swap point in
    ``model_wrapper`` before FSDP2 wrap. Handles both the per-expert ModuleList
    (Qwen3-MoE / Qwen3-Next today) and the fused ``gate_up_proj`` layout
    (transformers 5.0+, split at ``moe_dim = shape[1] // 2``).

  * ``convert_hf_layer_to_tt`` / ``convert_tt_layer_to_hf`` — state-dict-level
    per-layer remap (lifted ~verbatim from prime-rl) used by the CPU roundtrip
    parity test (G3b-3): ``hf → grouped → hf`` must be lossless.

Mapping (per expert ``j``):
    experts.{j}.gate_proj.weight  -> experts.w1[j]
    experts.{j}.up_proj.weight    -> experts.w3[j]
    experts.{j}.down_proj.weight  -> experts.w2[j]
Router:
    mlp.gate.weight               -> router.gate.weight
Shared expert (Qwen3-Next, optional):
    shared_expert.gate_proj.weight -> shared_expert.w1.weight
    shared_expert.up_proj.weight   -> shared_expert.w3.weight
    shared_expert.down_proj.weight -> shared_expert.w2.weight
    shared_expert_gate.weight      -> shared_expert_gate.weight
"""

from __future__ import annotations

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Live-module remap (used by the swap point)                                  #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def remap_hf_block_to_moe(hf_block, moe) -> None:
    """Copy weights from a live HF ``*SparseMoeBlock`` into a grouped ``MoE``.

    Both modules must already be built with matching dims. Copies in-place into
    ``moe``'s parameters (preserving device/dtype of the destination).
    """
    # Router gate.
    moe.router.gate.weight.copy_(hf_block.gate.weight)

    # Routed experts: per-expert ModuleList OR fused gate_up_proj.
    if hasattr(hf_block, "experts") and hasattr(hf_block.experts, "gate_up_proj"):
        # Fused layout (transformers 5.0+): gate_up_proj (num_experts, 2*moe_dim, dim).
        gate_up_proj = hf_block.experts.gate_up_proj
        down_proj = hf_block.experts.down_proj
        moe_dim = gate_up_proj.shape[1] // 2
        moe.experts.w1.copy_(gate_up_proj[:, :moe_dim, :])  # gate
        moe.experts.w3.copy_(gate_up_proj[:, moe_dim:, :])  # up
        moe.experts.w2.copy_(down_proj)  # down
    else:
        num_experts = len(hf_block.experts)
        for j in range(num_experts):
            expert = hf_block.experts[j]
            moe.experts.w1[j].copy_(expert.gate_proj.weight)
            moe.experts.w3[j].copy_(expert.up_proj.weight)
            moe.experts.w2[j].copy_(expert.down_proj.weight)

    # Shared expert (Qwen3-Next). Optional.
    if getattr(hf_block, "shared_expert", None) is not None and moe.shared_expert is not None:
        moe.shared_expert.w1.weight.copy_(hf_block.shared_expert.gate_proj.weight)
        moe.shared_expert.w3.weight.copy_(hf_block.shared_expert.up_proj.weight)
        moe.shared_expert.w2.weight.copy_(hf_block.shared_expert.down_proj.weight)
        if getattr(hf_block, "shared_expert_gate", None) is not None and moe.shared_expert_gate is not None:
            moe.shared_expert_gate.weight.copy_(hf_block.shared_expert_gate.weight)


# --------------------------------------------------------------------------- #
# State-dict-level remap (used by the CPU roundtrip test, G3b-3)               #
# --------------------------------------------------------------------------- #


def _get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    """Convert one layer's MoE weights HF → grouped, in place."""
    i = layer_idx
    gate_key = f"model.layers.{i}.mlp.gate.weight"
    if gate_key not in state_dict:
        return

    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict.pop(gate_key)

    if f"model.layers.{i}.mlp.experts.gate_up_proj" in state_dict:
        gate_up_proj = state_dict.pop(f"model.layers.{i}.mlp.experts.gate_up_proj")
        down_proj = state_dict.pop(f"model.layers.{i}.mlp.experts.down_proj")
        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]  # gate
        w3 = gate_up_proj[:, moe_dim:, :]  # up
        w2 = down_proj  # down
    else:
        num_experts = len(
            [k for k in state_dict.keys() if f"model.layers.{i}.mlp.experts" in k and "gate_proj" in k]
        )
        if num_experts == 0:
            return
        dim, moe_dim = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].shape
        dtype = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        for j in range(num_experts):
            w1[j].copy_(state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"))
            w2[j].copy_(state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"))
            w3[j].copy_(state_dict.pop(f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"))

    state_dict[f"model.layers.{i}.mlp.experts.w1"] = w1
    state_dict[f"model.layers.{i}.mlp.experts.w2"] = w2
    state_dict[f"model.layers.{i}.mlp.experts.w3"] = w3

    se_gate_key = f"model.layers.{i}.mlp.shared_expert.gate_proj.weight"
    if se_gate_key in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_expert.w1.weight"] = state_dict.pop(se_gate_key)
        state_dict[f"model.layers.{i}.mlp.shared_expert.w2.weight"] = state_dict.pop(
            f"model.layers.{i}.mlp.shared_expert.down_proj.weight"
        )
        state_dict[f"model.layers.{i}.mlp.shared_expert.w3.weight"] = state_dict.pop(
            f"model.layers.{i}.mlp.shared_expert.up_proj.weight"
        )
    # shared_expert_gate.weight is identical in both layouts — no rename needed.


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    """Convert one layer's MoE weights grouped → per-expert HF, in place."""
    i = layer_idx
    router_key = f"model.layers.{i}.mlp.router.gate.weight"
    if router_key not in state_dict:
        return

    state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict.pop(router_key)

    w1 = state_dict.pop(f"model.layers.{i}.mlp.experts.w1")
    w2 = state_dict.pop(f"model.layers.{i}.mlp.experts.w2")
    w3 = state_dict.pop(f"model.layers.{i}.mlp.experts.w3")
    num_experts = w1.shape[0]
    for j in range(num_experts):
        state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = w1[j]
        state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = w2[j]
        state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = w3[j]

    se_w1_key = f"model.layers.{i}.mlp.shared_expert.w1.weight"
    if se_w1_key in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_expert.gate_proj.weight"] = state_dict.pop(se_w1_key)
        state_dict[f"model.layers.{i}.mlp.shared_expert.down_proj.weight"] = state_dict.pop(
            f"model.layers.{i}.mlp.shared_expert.w2.weight"
        )
        state_dict[f"model.layers.{i}.mlp.shared_expert.up_proj.weight"] = state_dict.pop(
            f"model.layers.{i}.mlp.shared_expert.w3.weight"
        )


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert all MoE weights HF → grouped, in place. Returns the same dict."""
    for i in range(_get_max_layer_num(state_dict)):
        convert_hf_layer_to_tt(state_dict, i)
    return state_dict


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert all MoE weights grouped → per-expert HF, in place. Returns the same dict."""
    for i in range(_get_max_layer_num(state_dict)):
        convert_tt_layer_to_hf(state_dict, i)
    return state_dict
