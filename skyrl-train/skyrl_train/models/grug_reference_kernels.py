"""Gradients of the trainer's reference chains, computed without running the chains' forward.

Under ``Numerics.EXACT`` a region's value comes from compiled vLLM's kernel and its gradient is
the gradient of the trainer's chain for the same value (``grug_rounding``). The functions here return the kernel's value
with the gradient that autograd's backward of that chain gives, computed by one Triton kernel per backward from the
chain's bf16 inputs, without running the chain. Each kernel evaluates autograd's op-by-op formulas in their order and
roundings: every PyTorch elementwise op becomes the same fp32 operation (IEEE division, libdevice's exponential, no
fused multiply-add except where nvcc forms one in PyTorch's kernel), and each cast rounds where the chain casts.
``enable_reflect_ftz`` off keeps libdevice from flushing subnormals, which PyTorch's kernels keep.

- ``gated_product_value``: ``(normalized.float() * torch.sigmoid(gate.float())).to(bf16)``;
- ``swiglu_value``: ``(F.silu(gate.float()) * up.float()).to(bf16)`` of a fused ``[gate | up]`` projection;
- ``hybrid_input_norm_value``: ``(hidden.float() * torch.rsqrt(variance + eps) * weight.float()).to(bf16)`` with the
  variance differentiated as ``hidden.float().pow(2).mean(-1)``; its two broadcast reductions run as autograd's
  ``sum_to`` runs them, on the same fp32 products;
- ``xsa_head_gate_value``: XSA and the ``2 * sigmoid`` head gate in fp32 (``xsa_and_gate_single_rounding``), whose
  per-head reductions run as the same torch sums;
- ``query_key_values``: the q/k RMS norms, half RoPE with the bf16 table and the query scale in fp32
  (``rounded_query_key`` of ``qk_norm_fp32``), including the zero-padded slice gradients' sums that turn -0 into +0;
- ``router_logits_value``: ``F.linear(input.float(), weight.float())``, whose backward is two fp32 GEMMs that run
  as autograd issues them.
"""

from __future__ import annotations

import torch

from skyrl_train.models.grug_moe import GRUG_ATTN_GATE_SCALE, GRUG_QK_RMS_NORM_EPS, GRUG_XSA_EPS

_BLOCK = 1024
# The XSA and q/k gradient kernels index each head's elements in 128-element rows (Grug's head dimension).
_HEAD_DIM = 128
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

    @triton.jit
    def xsa_forward_terms(
        attention_ptr, value_ptr, products_ptr, squares_ptr, numel, GROUP: tl.constexpr, BLOCK: tl.constexpr
    ):
        """The chain's fp32 products ``attention * v`` and ``v.square()`` over ``[rows, heads, head_dim]``, with ``v`` the
        value head of each query head's group: element ``i`` of query head ``h`` reads element ``i`` of value head
        ``h // GROUP``."""
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
        value = tl.load(value_ptr + (offsets // 128 // GROUP) * 128 + offsets % 128, mask=mask).to(tl.float32)
        tl.store(products_ptr + offsets, attention * value, mask=mask)
        tl.store(squares_ptr + offsets, value * value, mask=mask)  # pow(v, 2) multiplies v by itself

    @triton.jit
    def xsa_backward_terms(
        dy_ptr,
        attention_ptr,
        value_ptr,
        quotient_ptr,
        scale_ptr,
        scale_terms_ptr,
        quotient_terms_ptr,
        numel,
        GROUP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        head_row = offsets // 128
        grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)  # ToCopyBackward of .to(bf16)
        attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
        value = tl.load(value_ptr + (offsets // 128 // GROUP) * 128 + offsets % 128, mask=mask).to(tl.float32)
        quotient = tl.load(quotient_ptr + head_row, mask=mask)
        scale = tl.load(scale_ptr + head_row, mask=mask)
        projected = attention - quotient * value  # the forward's attention - (dot / denominator) * v
        tl.store(
            scale_terms_ptr + offsets, grad * projected, mask=mask
        )  # MulBackward0 grad * self, summed to the scale
        d_projected = grad * scale  # MulBackward0 grad * other
        tl.store(quotient_terms_ptr + offsets, (-d_projected) * value, mask=mask)  # SubBackward0's -grad, times v

    @triton.jit
    def xsa_backward_inputs(
        dy_ptr,
        attention_ptr,
        value_ptr,
        quotient_ptr,
        scale_ptr,
        d_dot_ptr,
        d_denominator_ptr,
        d_attention_ptr,
        d_value_heads_ptr,
        numel,
        GROUP: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        head_row = offsets // 128
        grad = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)
        attention = tl.load(attention_ptr + offsets, mask=mask).to(tl.float32)
        value = tl.load(value_ptr + (offsets // 128 // GROUP) * 128 + offsets % 128, mask=mask).to(tl.float32)
        quotient = tl.load(quotient_ptr + head_row, mask=mask)
        scale = tl.load(scale_ptr + head_row, mask=mask)
        d_dot = tl.load(d_dot_ptr + head_row, mask=mask)
        d_denominator = tl.load(d_denominator_ptr + head_row, mask=mask)
        d_projected = grad * scale
        minus_d_projected = -d_projected
        from_quotient = minus_d_projected * quotient  # MulBackward0 of quotient * v: grad * self
        from_square = d_denominator * (2.0 * value)  # PowBackward0 of v.pow(2): grad * (2 * v.pow(1))
        from_dot_value = d_dot * attention  # MulBackward0 of attention * v: grad * self
        from_dot_attention = d_dot * value  # MulBackward0 of attention * v: grad * other
        # Autograd adds a tensor's gradients in the order their nodes run: the subtraction's before the dot's for the
        # attention, and for v the quotient product's, then the square's, then the dot's.
        d_attention = d_projected + from_dot_attention
        d_value = (from_quotient + from_square) + from_dot_value
        tl.store(d_attention_ptr + offsets, d_attention.to(tl.bfloat16), mask=mask)
        tl.store(d_value_heads_ptr + offsets, d_value.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _qk_scaled(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY: tl.constexpr):
        """The gradient reaching the RoPE output: the query's two scale multiplications run backward (scale first)."""
        grad = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)  # ToCopyBackward of .to(bf16)
        if QUERY:
            grad = (grad * multiplier_scale) * multiplier
        return grad

    @triton.jit
    def _qk_rotary_grad(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY: tl.constexpr):
        """The gradient of an element of the rotated half: the query's passes its bf16 round trip and the sum of the two
        zero-padded slice gradients (which turns -0 into +0); the key's reaches the rotation directly."""
        grad = _qk_scaled(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY)
        if QUERY:
            grad = grad.to(tl.bfloat16).to(tl.float32) + 0.0
        return grad

    @triton.jit
    def qk_backward_terms(
        dy_ptr,
        raw_ptr,
        rsqrt_ptr,
        cos_ptr,
        sin_ptr,
        scaled_ptr,
        products_ptr,
        numel,
        multiplier,
        multiplier_scale,
        HEADS_PER_POSITION: tl.constexpr,
        ROTARY: tl.constexpr,
        QUERY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        head_row = offsets // 128
        lane = offsets % 128
        if ROTARY:
            position = head_row // HEADS_PER_POSITION
            rotary = lane < 64
            rotary_lane = tl.where(rotary, lane, 0)
            partner = tl.where(rotary_lane < 32, rotary_lane + 32, rotary_lane - 32)
            row_start = head_row * 128
            rotary_mask = mask & rotary
            own = _qk_rotary_grad(dy_ptr, row_start + rotary_lane, rotary_mask, multiplier, multiplier_scale, QUERY)
            other = _qk_rotary_grad(dy_ptr, row_start + partner, rotary_mask, multiplier, multiplier_scale, QUERY)
            cos = tl.load(cos_ptr + position * 64 + rotary_lane, mask=rotary_mask, other=0.0)
            sin = tl.load(sin_ptr + position * 64 + partner, mask=rotary_mask, other=0.0)
            # Rotation backward: grad * cos, plus the partner's grad * sin (negated for the second half), the multiply's
            # and the chunk's gradients of the rotary slice added.
            from_partner = other * sin
            rotated = own * cos + tl.where(rotary_lane < 32, from_partner, -from_partner)
            passthrough = _qk_scaled(dy_ptr, offsets, mask & (lane >= 64), multiplier, multiplier_scale, QUERY)
            grad = tl.where(rotary, rotated, passthrough) + 0.0  # the slices' zero-padded gradients summed
        else:
            grad = _qk_scaled(dy_ptr, offsets, mask, multiplier, multiplier_scale, QUERY)
        raw = tl.load(raw_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(rsqrt_ptr + head_row, mask=mask, other=0.0)
        tl.store(scaled_ptr + offsets, grad * rsqrt, mask=mask)  # MulBackward0 of fp32 * rsqrt: grad * other
        tl.store(products_ptr + offsets, grad * raw, mask=mask)  # grad * self, summed to the rsqrt

    @triton.jit
    def qk_backward_inputs(scaled_ptr, raw_ptr, d_mean_ptr, d_raw_ptr, numel, BLOCK: tl.constexpr):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        raw = tl.load(raw_ptr + offsets, mask=mask).to(tl.float32)
        d_square = tl.load(d_mean_ptr + offsets // 128, mask=mask) * 0.0078125  # MeanBackward1: grad * (1 / 128)
        from_square = d_square * (2.0 * raw)  # PowBackward0: grad * (2 * self.pow(1))
        d_raw = tl.load(scaled_ptr + offsets, mask=mask) + from_square
        tl.store(d_raw_ptr + offsets, d_raw.to(tl.bfloat16), mask=mask)  # ToCopyBackward of .float()

    _COMPILED = (
        gated_product_backward,
        swiglu_backward,
        hybrid_norm_backward_terms,
        variance_backward,
        xsa_forward_terms,
        xsa_backward_terms,
        xsa_backward_inputs,
        qk_backward_terms,
        qk_backward_inputs,
    )
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
    """``value`` differentiated as ``rms_norm_hybrid(hidden, variance_with_gradient(variance, hidden), weight, eps)``.

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


class _XsaHeadGate(torch.autograd.Function):
    """``value`` differentiated as ``xsa_and_gate_single_rounding(attention, kv_value, gate, head_dim)``.

    The chain's reductions over each head's ``head_dim`` values (the dot product and the value's squared norm in the
    forward, the scale's and the quotient's broadcast gradients in the backward) run as the same torch sums on the same
    fp32 products; the repeated value's gradient is summed over each group's query heads as the expand's ``sum_to`` sums
    it.
    """

    @staticmethod
    def forward(ctx, value, attention, kv_value, gate, head_dim):
        ctx.save_for_backward(attention, kv_value, gate)
        ctx.head_dim = head_dim
        return value

    @staticmethod
    def backward(ctx, grad):
        attention, kv_value, gate = (tensor.contiguous() for tensor in ctx.saved_tensors)
        grad = grad.contiguous()
        head_dim = ctx.head_dim
        if head_dim != _HEAD_DIM:
            raise NotImplementedError(f"the XSA gradient kernels address {_HEAD_DIM}-dim heads, got {head_dim}")
        rows = kv_value.shape[:-2]
        kv_heads = kv_value.shape[-2]
        heads = attention.shape[-1] // head_dim
        group = heads // kv_heads
        numel = attention.numel()
        kernels = _kernels()
        products = torch.empty(*rows, heads, head_dim, dtype=torch.float32, device=attention.device)
        squares = torch.empty_like(products)
        kernels[4][_grid(numel)](
            attention, kv_value, products, squares, numel, GROUP=group, BLOCK=_BLOCK, **_EXACT_LAUNCH
        )
        dot = products.sum(dim=-1, keepdim=True)
        denominator = squares.sum(dim=-1, keepdim=True) + GRUG_XSA_EPS
        quotient = dot / denominator
        sigmoid = torch.sigmoid(gate.float())
        scale = GRUG_ATTN_GATE_SCALE * sigmoid
        scale_terms, quotient_terms = torch.empty_like(products), torch.empty_like(products)
        kernels[5][_grid(numel)](
            grad,
            attention,
            kv_value,
            quotient,
            scale,
            scale_terms,
            quotient_terms,
            numel,
            GROUP=group,
            BLOCK=_BLOCK,
            **_EXACT_LAUNCH,
        )
        d_scale = scale_terms.sum(dim=-1, keepdim=True)
        d_quotient = quotient_terms.sum(dim=-1, keepdim=True)
        d_gate = torch.ops.aten.sigmoid_backward(d_scale.view(gate.shape) * GRUG_ATTN_GATE_SCALE, sigmoid)
        d_dot = d_quotient / denominator  # DivBackward0: grad / other
        d_denominator = -d_quotient * (
            (dot / denominator) / denominator
        )  # DivBackward0: -grad * ((self / other) / other)
        d_attention = torch.empty_like(attention)
        d_value_heads = torch.empty(*rows, heads, head_dim, dtype=kv_value.dtype, device=kv_value.device)
        kernels[6][_grid(numel)](
            grad,
            attention,
            kv_value,
            quotient,
            scale,
            d_dot,
            d_denominator,
            d_attention,
            d_value_heads,
            numel,
            GROUP=group,
            BLOCK=_BLOCK,
            **_EXACT_LAUNCH,
        )
        # repeat_interleave is an expand of the unsqueezed value: its backward sums each group's heads (sum_to).
        d_value = d_value_heads.view(*rows, kv_heads, group, head_dim).sum([len(rows) + 1], keepdim=True)
        return None, d_attention, d_value.view(kv_value.shape), d_gate.to(gate.dtype), None


class _QueryKeyReference(torch.autograd.Function):
    """``value`` differentiated as one output of ``rounded_query_key(qk_norm_fp32(query), qk_norm_fp32(key), ...)``.

    The query's: ``rotate_neox_fp32`` on its first 64 dims (sliding-window layers), the rotated half's bf16 round trip,
    then ``* multiplier * multiplier_scale`` and the bf16 cast. The key's: the rotation and the cast. Each norm's
    reduction (the mean of squares in the forward, the rsqrt's broadcast gradient in the backward) runs as the same
    torch op on the same fp32 tensor.
    """

    @staticmethod
    def forward(ctx, value, raw, freqs, multiplier, multiplier_scale, query):
        ctx.rotary = freqs is not None
        ctx.save_for_backward(raw, *((freqs,) if ctx.rotary else ()))
        ctx.multipliers = (multiplier, multiplier_scale)
        ctx.query = query
        return value

    @staticmethod
    def backward(ctx, grad):
        raw, *rest = ctx.saved_tensors
        raw, grad = raw.contiguous(), grad.contiguous()
        if raw.shape[-1] != _HEAD_DIM:
            raise NotImplementedError(f"the q/k gradient kernels address {_HEAD_DIM}-dim heads")
        rsqrt = torch.rsqrt(raw.float().square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS)
        if ctx.rotary:
            (freqs,) = rest
            if freqs.shape[-1] != 64:
                raise NotImplementedError("the q/k gradient kernels rotate 64 of each head's 128 dims")
            cos = torch.cos(freqs).to(torch.bfloat16).float().contiguous()
            sin = torch.sin(freqs).to(torch.bfloat16).float().contiguous()
        else:
            cos = sin = rsqrt  # unread
        numel = raw.numel()
        scaled = torch.empty(raw.shape, dtype=torch.float32, device=raw.device)
        products = torch.empty_like(scaled)
        kernels = _kernels()
        kernels[7][_grid(numel)](
            grad,
            raw,
            rsqrt.contiguous(),
            cos,
            sin,
            scaled,
            products,
            numel,
            *ctx.multipliers,
            HEADS_PER_POSITION=raw.shape[-3] * raw.shape[-2] if raw.dim() == 4 else raw.shape[-2],
            ROTARY=ctx.rotary,
            QUERY=ctx.query,
            BLOCK=_BLOCK,
            **_EXACT_LAUNCH,
        )
        d_rsqrt = products.sum(dim=-1, keepdim=True)
        d_mean = (-0.5 * d_rsqrt) * rsqrt.pow(3)  # RsqrtBackward0; the eps add passes the gradient through
        d_raw = torch.empty_like(raw)
        kernels[8][_grid(numel)](scaled, raw, d_mean.contiguous(), d_raw, numel, BLOCK=_BLOCK, **_EXACT_LAUNCH)
        return None, d_raw, None, None, None, None


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
    """``value``, differentiated as ``rms_norm_hybrid(hidden, variance_with_gradient(variance, hidden), weight, eps)``:
    the bf16 input normalized by the unrounded sum's variance, whose gradient goes through ``hidden``'s variance."""
    if not torch.is_grad_enabled():
        return value
    return _HybridInputNorm.apply(value, hidden, hidden, variance, weight, eps)


def xsa_head_gate_value(
    value: torch.Tensor, attention: torch.Tensor, kv_value: torch.Tensor, gate: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """``value``, differentiated as ``xsa_and_gate_single_rounding(attention, kv_value, gate, head_dim)``."""
    if not torch.is_grad_enabled():
        return value
    return _XsaHeadGate.apply(value, attention, kv_value, gate, head_dim)


def query_key_values(
    query_value: torch.Tensor,
    key_value: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor] | None,
    multiplier: float,
    multiplier_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The vLLM query and key values, differentiated as ``rounded_query_key(qk_norm_fp32(query), qk_norm_fp32(key),
    freqs, multiplier, multiplier_scale, dtype)`` with ``freqs`` the query's and the key's rotary angles (``None``
    without RoPE)."""
    if not torch.is_grad_enabled():
        return query_value, key_value
    query_freqs, key_freqs = freqs if freqs is not None else (None, None)
    return (
        _QueryKeyReference.apply(query_value, query, query_freqs, multiplier, multiplier_scale, True),
        _QueryKeyReference.apply(key_value, key, key_freqs, 1.0, 1.0, False),
    )


def router_logits_value(value: torch.Tensor, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``value``, differentiated as ``F.linear(input.float(), weight.float())``."""
    if not torch.is_grad_enabled():
        return value
    return _RouterLogits.apply(value, input, weight)
