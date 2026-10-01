"""Grug arithmetic with compiled vLLM's rounding points: fp32 through a fused chain, one bf16 rounding.

Used by the Megatron Grug modules when a ``mismatch_probe.numerics`` flag is active.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_XSA_EPS


class _ValueWithGradient(torch.autograd.Function):
    """The first tensor's bytes, differentiated as the second: the trainer's own computation of the same value."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, grad


def vllm_value(value: torch.Tensor, reference: Callable[[], torch.Tensor]) -> torch.Tensor:
    """A vLLM kernel's ``value``; with gradients enabled, differentiated as the trainer's ``reference()``.

    ``reference()`` computes the same values up to rounding with an autograd graph; it runs only when gradients are
    enabled. The result holds ``value``'s bytes exactly (no arithmetic touches them).
    """
    if not torch.is_grad_enabled():
        return value
    return _ValueWithGradient.apply(value, reference())


def rms_norm_single_rounding(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM's RMSNorm on an fp32 input: ``x * rsqrt(mean(x^2) + eps) * weight`` in fp32, rounded once."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * weight.float()).to(weight.dtype)


def rms_norm_hybrid(rounded: torch.Tensor, variance: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compiled vLLM's input norm: the stored bf16 sum normalized by the variance of the unrounded sum."""
    return (rounded.float() * torch.rsqrt(variance + eps) * weight.float()).to(weight.dtype)


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


def weighted_down_projection_single_rounding(
    activation: torch.Tensor, weights: list[torch.Tensor], tokens_per_expert: list[int], probs: torch.Tensor
) -> torch.Tensor:
    """Per-expert ``activation @ W.T`` accumulated in fp32, times the route weight, rounded once.

    ``activation`` rows are grouped by expert in ``tokens_per_expert`` order; ``probs`` is ``[rows, 1]``.
    """
    outputs = [
        rows.float() @ weight.float().t()
        for rows, weight in zip(activation.split(tokens_per_expert), weights, strict=True)
    ]
    return (torch.cat(outputs) * probs.float()).to(activation.dtype)


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
