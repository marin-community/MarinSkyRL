"""Grug arithmetic with compiled vLLM's rounding points: fp32 through a fused chain, one bf16 rounding.

Used by the Megatron Grug modules when a ``mismatch_probe.numerics`` flag is active.
"""

import torch
import torch.nn.functional as F

from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_XSA_EPS


def gated_norm_product_fp32(normalized: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``norm(x) * sigmoid(gate)`` in fp32 from the bf16 norm output and gate projection, unrounded."""
    return normalized.float() * torch.sigmoid(gate.float())


def rotate_neox_fp32(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE on the leading rotary dims in fp32, with the bf16 cos/sin table compiled vLLM reads.

    ``freqs`` is Megatron's ``[seq, 1, 1, rotary_dim]`` angle tensor (angles repeated for each half).
    """
    rotary_dim = freqs.shape[-1]
    rotary, passthrough = tensor[..., :rotary_dim].float(), tensor[..., rotary_dim:].float()
    cos = torch.cos(freqs).to(torch.bfloat16).float()
    sin = torch.sin(freqs).to(torch.bfloat16).float()
    first, second = rotary.chunk(2, dim=-1)
    rotated = rotary * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((rotated, passthrough), dim=-1)


def swiglu_single_rounding(fc1_output: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` in fp32 from a fused ``[gate | up]`` projection, rounded once."""
    gate, up = torch.chunk(fc1_output, 2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(fc1_output.dtype)


def xsa_and_gate_single_rounding(
    core_attn_out: torch.Tensor, value: torch.Tensor, gate: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """XSA then the ``2 * sigmoid`` head gate in fp32, rounded once.

    ``core_attn_out`` is ``[..., heads * head_dim]``, ``value`` is ``[..., kv_heads, head_dim]`` and
    ``gate`` is ``[..., heads]``; query heads map to KV heads in contiguous groups.
    """
    heads = core_attn_out.shape[-1] // head_dim
    attention = core_attn_out.view(*value.shape[:-2], heads, head_dim).float()
    v = value.repeat_interleave(heads // value.shape[-2], dim=-2).float()
    dot = (attention * v).sum(dim=-1, keepdim=True)
    projected = attention - (dot / (v.square().sum(dim=-1, keepdim=True) + GRUG_XSA_EPS)) * v
    scale = (GRUG_ATTN_GATE_SCALE * torch.sigmoid(gate.float())).view(*gate.shape, 1)
    return (projected * scale).to(core_attn_out.dtype).reshape(core_attn_out.shape)

