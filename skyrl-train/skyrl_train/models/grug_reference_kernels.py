"""Gradients of the trainer's reference chains, computed without running the chains' forward.

Under the vLLM-kernel numerics a region's value comes from compiled vLLM's kernel and its gradient from the trainer's
own computation of the same value (``grug_rounding.vllm_value``): with gradients enabled the trainer's chain runs only to
build the autograd graph its backward walks. The functions here return the kernel's value with the gradient that
autograd's backward of that chain gives, computed by one Triton kernel per backward from the chain's bf16 inputs. Each
kernel evaluates autograd's op-by-op formulas in their order and roundings: every PyTorch elementwise op becomes the
same fp32 operation (IEEE division, libdevice's exponential, no fused multiply-add except where nvcc forms one in
PyTorch's kernel), and each cast rounds where the chain casts. ``enable_reflect_ftz`` off keeps libdevice from flushing
subnormals, which PyTorch's kernels keep.

- ``gated_product_value``: ``(normalized.float() * torch.sigmoid(gate.float())).to(bf16)``;
- ``swiglu_value``: ``(F.silu(gate.float()) * up.float()).to(bf16)`` of a fused ``[gate | up]`` projection;
- ``hybrid_input_norm_value``: ``(hidden.float() * torch.rsqrt(variance + eps) * weight.float()).to(bf16)`` with the
  variance differentiated as ``hidden.float().pow(2).mean(-1)``; its two broadcast reductions run as autograd's
  ``sum_to`` runs them, on the same fp32 products;
- ``router_logits_value``: ``F.linear(input.float(), weight.float())``, whose backward is two fp32 GEMMs that run
  as autograd issues them.
"""

from __future__ import annotations

import torch

_BLOCK = 1024
_EXACT_LAUNCH = {"enable_fp_fusion": False, "enable_reflect_ftz": False, "num_warps": 4}


def _kernels():
    """The Triton kernels, compiled on first use (Triton is not in the CPU test lane)."""
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def gated_product_backward(
        dy_ptr, normalized_ptr, gate_ptr, d_normalized_ptr, d_gate_ptr, numel, BLOCK: tl.constexpr
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
        normalized = tl.load(normalized_ptr + offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))  # torch.sigmoid's saved result
        d_normalized = grad * sigmoid  # MulBackward0: grad * other
        d_sigmoid = grad * normalized  # MulBackward0: grad * self
        d_gate = (d_sigmoid * (1.0 - sigmoid)) * sigmoid  # sigmoid_backward: a * (1 - b) * b
        tl.store(d_normalized_ptr + offsets, d_normalized.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()
        tl.store(d_gate_ptr + offsets, d_gate.to(tl.bfloat16), mask=mask)

    @triton.jit
    def swiglu_backward(dy_ptr, fc1_ptr, d_fc1_ptr, numel, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        gate_offsets = (offsets // WIDTH) * (2 * WIDTH) + offsets % WIDTH
        grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
        gate = tl.load(fc1_ptr + gate_offsets, mask=mask).to(tl.float32)
        up = tl.load(fc1_ptr + gate_offsets + WIDTH, mask=mask).to(tl.float32)
        silu = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate))  # F.silu's saved result: x / (1 + exp(-x))
        d_silu = grad * up  # MulBackward0: grad * other
        d_up = grad * silu  # MulBackward0: grad * self
        sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        # silu_backward: dy * s * (1 + x * (1 - s)), where nvcc fuses 1 + x * (1 - s) into one multiply-add.
        d_gate = (d_silu * sigmoid) * libdevice.fma(gate, 1.0 - sigmoid, 1.0)
        tl.store(d_fc1_ptr + gate_offsets, d_gate.to(tl.bfloat16), mask=mask)  # chunk's backward concatenates
        tl.store(d_fc1_ptr + gate_offsets + WIDTH, d_up.to(tl.bfloat16), mask=mask)

    @triton.jit
    def hybrid_norm_backward_terms(
        dy_ptr,
        hidden_ptr,
        weight_ptr,
        rsqrt_ptr,
        d_hidden_ptr,
        weight_terms_ptr,
        rsqrt_terms_ptr,
        numel,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
        hidden = tl.load(hidden_ptr + offsets, mask=mask).to(tl.float32)
        weight = tl.load(weight_ptr + offsets % WIDTH, mask=mask).to(tl.float32)
        rsqrt = tl.load(rsqrt_ptr + offsets // WIDTH, mask=mask)
        normalized = hidden * rsqrt  # the forward's rounded.float() * rsqrt(variance + eps)
        d_normalized = grad * weight  # MulBackward0 of normalized * weight.float(): grad * other
        tl.store(weight_terms_ptr + offsets, grad * normalized, mask=mask)  # grad * self, summed to the weight
        d_hidden = d_normalized * rsqrt  # MulBackward0 of rounded.float() * rsqrt: grad * other
        tl.store(rsqrt_terms_ptr + offsets, d_normalized * hidden, mask=mask)  # grad * self, summed to the rsqrt
        tl.store(d_hidden_ptr + offsets, d_hidden.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()

    @triton.jit
    def variance_backward(
        d_variance_ptr, hidden_ptr, d_hidden_ptr, inverse_width, numel, WIDTH: tl.constexpr, BLOCK: tl.constexpr
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        hidden = tl.load(hidden_ptr + offsets, mask=mask).to(tl.float32)
        d_variance = tl.load(d_variance_ptr + offsets // WIDTH, mask=mask)
        d_square = d_variance * inverse_width  # MeanBackward1: grad.expand(...) / width, as grad * (1 / width)
        d_hidden = d_square * (2.0 * hidden)  # PowBackward0: grad * (2 * self.pow(1))
        tl.store(d_hidden_ptr + offsets, d_hidden.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()

    _COMPILED = (gated_product_backward, swiglu_backward, hybrid_norm_backward_terms, variance_backward)
    return _COMPILED


_COMPILED = None


def _grid(numel: int) -> tuple[int]:
    return (-(-numel // _BLOCK),)


class _GatedProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, normalized, gate):
        ctx.save_for_backward(normalized, gate)
        return value

    @staticmethod
    def backward(ctx, grad):
        normalized, gate = ctx.saved_tensors
        grad = grad.contiguous()
        normalized, gate = normalized.contiguous(), gate.contiguous()
        d_normalized, d_gate = torch.empty_like(normalized), torch.empty_like(gate)
        gated_product_backward = _kernels()[0]
        gated_product_backward[_grid(grad.numel())](
            grad, normalized, gate, d_normalized, d_gate, grad.numel(), BLOCK=_BLOCK, **_EXACT_LAUNCH
        )
        return None, d_normalized, d_gate


class _Swiglu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, fc1_output):
        ctx.save_for_backward(fc1_output)
        return value

    @staticmethod
    def backward(ctx, grad):
        (fc1_output,) = ctx.saved_tensors
        grad, fc1_output = grad.contiguous(), fc1_output.contiguous()
        d_fc1 = torch.empty_like(fc1_output)
        swiglu_backward = _kernels()[1]
        swiglu_backward[_grid(grad.numel())](
            grad, fc1_output, d_fc1, grad.numel(), WIDTH=grad.shape[-1], BLOCK=_BLOCK, **_EXACT_LAUNCH
        )
        return None, d_fc1


class _HybridInputNorm(torch.autograd.Function):
    """``value`` differentiated as ``rms_norm_hybrid(hidden, _variance_with_gradient(variance, hidden), weight, eps)``.

    ``hidden`` comes in twice, as the chain reads it twice (``hidden.float()`` in the norm and in the variance's
    reference); the backward hands each read its own gradient, in the order autograd's engine delivers them (the norm's
    read first: its node is the later one in the forward), so their sum with the layer's other uses of ``hidden`` adds
    in the same order.
    """

    @staticmethod
    def forward(ctx, value, hidden, hidden_again, variance, weight, eps):
        ctx.save_for_backward(hidden, variance, weight)
        ctx.eps = eps
        return value

    @staticmethod
    def backward(ctx, grad):
        hidden, variance, weight = ctx.saved_tensors
        grad, hidden = grad.contiguous(), hidden.contiguous()
        width = hidden.shape[-1]
        rsqrt = torch.rsqrt(variance + ctx.eps)  # the forward's RsqrtBackward0 result, recomputed by the same ops
        d_hidden = torch.empty_like(hidden)
        weight_terms = torch.empty(hidden.shape, dtype=torch.float32, device=hidden.device)
        rsqrt_terms = torch.empty_like(weight_terms)
        kernels = _kernels()
        kernels[2][_grid(grad.numel())](
            grad,
            hidden,
            weight,
            rsqrt.contiguous(),
            d_hidden,
            weight_terms,
            rsqrt_terms,
            grad.numel(),
            WIDTH=width,
            BLOCK=_BLOCK,
            **_EXACT_LAUNCH,
        )
        # The broadcast operands' gradients, reduced as autograd's sum_to reduces them.
        d_weight = weight_terms.sum(list(range(hidden.dim() - 1)), keepdim=True).view(weight.shape).to(weight.dtype)
        d_rsqrt = rsqrt_terms.sum([hidden.dim() - 1], keepdim=True)
        d_variance = (-0.5 * d_rsqrt) * rsqrt.pow(3)  # RsqrtBackward0; the eps add passes the gradient through
        d_hidden_variance = torch.empty_like(hidden)
        kernels[3][_grid(grad.numel())](
            d_variance.contiguous(),
            hidden,
            d_hidden_variance,
            float(1.0 / width),
            grad.numel(),
            WIDTH=width,
            BLOCK=_BLOCK,
            **_EXACT_LAUNCH,
        )
        return None, d_hidden, d_hidden_variance, None, d_weight, None


class _RouterLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, input, weight):
        ctx.save_for_backward(input, weight)
        return value

    @staticmethod
    def backward(ctx, grad):
        input, weight = ctx.saved_tensors
        # Autograd of F.linear(input.float(), weight.float()) on [S, B, K] input: matmul folds the input to
        # [S * B, K] rows and runs mm against weight.t(), a column-major view; MmBackward0 then issues
        # grad.mm(weight) for the rows and, for the column-major operand, grad.t().mm(rows).t(), which TBackward0
        # transposes back; each ToCopyBackward0 rounds to bf16.
        rows = input.reshape(-1, input.shape[-1]).float()
        grad_rows = grad.reshape(-1, grad.shape[-1])
        d_input = grad_rows.mm(weight.float()).view(input.shape).to(input.dtype)
        d_weight = grad_rows.t().mm(rows).to(weight.dtype)
        return None, d_input, d_weight


def gated_product_value(value: torch.Tensor, normalized: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``value``, differentiated as ``(normalized.float() * torch.sigmoid(gate.float())).to(bf16)``."""
    if not torch.is_grad_enabled():
        return value
    return _GatedProduct.apply(value, normalized, gate)


def swiglu_value(value: torch.Tensor, fc1_output: torch.Tensor) -> torch.Tensor:
    """``value``, differentiated as ``(F.silu(gate.float()) * up.float()).to(bf16)`` of ``fc1_output = [gate | up]``."""
    if not torch.is_grad_enabled():
        return value
    return _Swiglu.apply(value, fc1_output)


def hybrid_input_norm_value(
    value: torch.Tensor, hidden: torch.Tensor, variance: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """``value``, differentiated as ``rms_norm_hybrid(hidden, _variance_with_gradient(variance, hidden), weight, eps)``:
    the bf16 input normalized by the unrounded sum's variance, whose gradient goes through ``hidden``'s variance."""
    if not torch.is_grad_enabled():
        return value
    return _HybridInputNorm.apply(value, hidden, hidden, variance, weight, eps)


def router_logits_value(value: torch.Tensor, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``value``, differentiated as ``F.linear(input.float(), weight.float())``."""
    if not torch.is_grad_enabled():
        return value
    return _RouterLogits.apply(value, input, weight)
