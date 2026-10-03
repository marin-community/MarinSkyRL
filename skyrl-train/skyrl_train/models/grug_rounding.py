"""Grug arithmetic with compiled vLLM's rounding points: fp32 through a fused chain, one bf16 rounding.

Under ``Numerics.EXACT`` a region's value comes from compiled vLLM's kernel (``vllm_value``) and its gradient is the
gradient of the region's chain here, which ``grug_reference_kernels`` computes without running the chain.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_QK_RMS_NORM_EPS, GRUG_XSA_EPS


class _ValueWithGradient(torch.autograd.Function):
    """The first tensor's bytes, differentiated as the second: the trainer's own computation of the same value."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Returned as is, autograd wraps the value in a view that carries this function's gradient: no copy.
        return value

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, grad


def vllm_value(value: torch.Tensor, reference: Callable[[], torch.Tensor]) -> torch.Tensor:
    """A vLLM kernel's ``value``; with gradients enabled, differentiated as the trainer's ``reference()``.

    ``reference()`` computes the same values up to rounding and runs only when gradients are enabled; the result holds
    ``value``'s bytes exactly.
    """
    if not torch.is_grad_enabled():
        return value
    return _ValueWithGradient.apply(value, reference())


# The extra columns a pipeline stage's output carries: one fp32 input-norm statistic per token, as two bf16 words.
STAGE_STATISTIC_COLUMNS = 2


class _AppendStageStatistic(torch.autograd.Function):
    """``[S, B, H]`` bf16 hidden states with each token's fp32 statistic appended as two bf16 words: ``[S, B, H + 2]``.

    The words are the statistic's bytes, moved without arithmetic. The statistic takes no gradient; the hidden states
    take the gradient of their columns.
    """

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
        ctx.width = hidden.shape[-1]
        words = statistic.float().contiguous().view(torch.int16)
        return torch.cat((hidden.contiguous().view(torch.int16), words), dim=-1).view(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad[..., : ctx.width], None


class _SplitStageStatistic(torch.autograd.Function):
    """The inverse of ``_AppendStageStatistic``: the ``[S, B, H]`` hidden states and ``[S, B, 1]`` fp32 statistic."""

    @staticmethod
    def forward(ctx, packed: torch.Tensor, width: int):
        ctx.packed_width = packed.shape[-1]
        words = packed.view(torch.int16)
        hidden = words[..., :width].contiguous().view(torch.bfloat16)
        statistic = words[..., width:].contiguous().view(torch.float32)
        ctx.mark_non_differentiable(statistic)
        return hidden, statistic

    @staticmethod
    def backward(ctx, grad_hidden: torch.Tensor, grad_statistic: torch.Tensor | None):
        grad = grad_hidden.new_zeros(*grad_hidden.shape[:-1], ctx.packed_width)
        grad[..., : grad_hidden.shape[-1]] = grad_hidden
        return grad, None


def append_stage_statistic(hidden: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
    return _AppendStageStatistic.apply(hidden, statistic)


def split_stage_statistic(packed: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _SplitStageStatistic.apply(packed, width)


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


def qk_norm_fp32(hidden_states: torch.Tensor) -> torch.Tensor:
    """Grug's weightless q/k RMS norm per head in fp32, unrounded."""
    fp32 = hidden_states.float()
    return fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS)


def rounded_query_key(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor] | None,
    multiplier: float,
    multiplier_scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled vLLM's rounding points for the fp32-normalized q and k: RoPE with the bf16 table, then the scale.

    ``freqs`` holds the query's and the key's rotary angles, ``None`` on a layer without RoPE. k rounds once. q's
    rotated half is stored once before the scale and rounds again after it; its pass-through half rounds once.
    """
    if freqs is not None:
        query_freqs, key_freqs = freqs
        rotary_dim = query_freqs.shape[-1]
        query = rotate_neox_fp32(query, query_freqs)
        query = torch.cat((query[..., :rotary_dim].to(dtype).float(), query[..., rotary_dim:]), dim=-1)
        key = rotate_neox_fp32(key, key_freqs)
    return (query.float() * multiplier * multiplier_scale).to(dtype), key.to(dtype)


def variance_with_gradient(variance: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """The unrounded sum's ``variance``, differentiated as the variance of the bf16 input it was rounded to."""
    return vllm_value(variance, lambda: hidden_states.float().pow(2).mean(dim=-1, keepdim=True))
