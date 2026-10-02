import pytest
import torch
import torch.nn.functional as F

from skyrl_train.models.grug_invariant_kernels import invariant_router_logits
from skyrl_train.models.grug_reference_kernels import (
    gated_product_value,
    hybrid_input_norm_value,
    query_key_values,
    router_logits_value,
    swiglu_value,
    xsa_head_gate_value,
)
from skyrl_train.models.grug_rounding import (
    gated_norm_product_fp32,
    qk_norm_fp32,
    rms_norm_hybrid,
    rounded_query_key,
    swiglu_single_rounding,
    variance_with_gradient,
    vllm_value,
    xsa_and_gate_single_rounding,
)
from tests.gpu.grug_gpu_gates import require_hoppers

# 3,328 rows put every non-NaN bf16 bit pattern into the head-gate logits ([rows, 1, 20]).
ROWS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, EXPERTS = 3328, 2560, 20, 5, 128, 256
QUERY_MULTIPLIER = 1.5703274004183787


def _every_bf16(shape, generator) -> torch.Tensor:
    """bf16 values holding every non-NaN bit pattern (then random ones), shuffled into ``shape``."""
    patterns = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    patterns = patterns[~patterns.float().isnan()]
    count = int(torch.tensor(shape).prod())
    picks = patterns[torch.randint(0, patterns.numel(), (count,), generator=generator)]
    picks[: min(count, patterns.numel())] = patterns[: min(count, patterns.numel())]
    return picks[torch.randperm(count, generator=generator)].view(shape).cuda()


def _activations(shape, generator) -> torch.Tensor:
    """bf16 values from fp32-subnormal to 1e3 in magnitude, with runs of +0 and -0."""
    scale = 10.0 ** torch.empty(shape).uniform_(-42, 3, generator=generator)
    values = (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)
    values.view(-1)[:64] = 0.0
    values.view(-1)[64:128] = -0.0
    return values.cuda()


def _leaves(*tensors):
    return [tensor.detach().clone().requires_grad_() for tensor in tensors]


def _assert_same_bytes(eager, fused):
    for eager_grad, fused_grad in zip(eager, fused, strict=True):
        width = torch.int16 if eager_grad.dtype == torch.bfloat16 else torch.int32
        assert torch.equal(eager_grad.contiguous().view(width), fused_grad.contiguous().view(width))


def _gated_product(generator):
    normalized, gate = _activations((ROWS, 1, HIDDEN), generator), _every_bf16((ROWS, 1, HIDDEN), generator)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        n, g = _leaves(normalized, gate)
        if fused:
            out = gated_product_value(value, n, g)
        else:
            out = vllm_value(value, lambda n=n, g=g: gated_norm_product_fp32(n, g).to(torch.bfloat16))
        out.backward(grad)
        results.append((n.grad, g.grad))
    return results


def _swiglu(generator):
    fc1 = torch.cat((_every_bf16((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)), dim=-1)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        (f,) = _leaves(fc1)
        out = swiglu_value(value, f) if fused else vllm_value(value, lambda f=f: swiglu_single_rounding(f))
        out.backward(grad)
        results.append((f.grad,))
    return results


def _router(generator):
    router_input = _activations((ROWS, 1, HIDDEN), generator)
    weight = (torch.randn(EXPERTS, HIDDEN, generator=generator) * 0.02).to(torch.bfloat16).cuda()
    value = torch.randn(ROWS, 1, EXPERTS, generator=generator).cuda()
    grad = torch.randn(ROWS, 1, EXPERTS, generator=generator).cuda()
    results = []
    for fused in (False, True):
        x, w = _leaves(router_input, weight)
        if fused:
            out = router_logits_value(value, x, w)
        else:
            out = vllm_value(value, lambda x=x, w=w: F.linear(x.float(), w.float()))
        out.backward(grad)
        results.append((x.grad, w.grad))
    return results


def _hybrid_input_norm(generator):
    # The layer input feeds the norm, the variance's reference and the residual add after the attention, so its
    # gradient adds three contributions; the order autograd adds them in is part of the check.
    hidden, weight = _activations((ROWS, 1, HIDDEN), generator), _activations((HIDDEN,), generator)
    variance = (torch.rand(ROWS, 1, 1, generator=generator) * 3 + 1e-3).cuda()
    variance[:4] = torch.tensor([1e-30, 3e-38, 5e-6, 1e30]).view(4, 1, 1)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    attention, residual_grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        h, w = _leaves(hidden, weight)
        if fused:
            normalized = hybrid_input_norm_value(value, h, variance, w, 1e-5)
        else:
            normalized = vllm_value(
                value, lambda h=h, w=w: rms_norm_hybrid(h, variance_with_gradient(variance, h), w, 1e-5)
            )
        torch.autograd.backward((normalized, h + attention), (grad, residual_grad))
        results.append((h.grad, w.grad))
    return results


def _xsa_head_gate(generator):
    attention = _activations((ROWS, 1, HEADS * HEAD_DIM), generator)
    kv_value = _activations((ROWS, 1, KV_HEADS, HEAD_DIM), generator)
    gate = _every_bf16((ROWS, 1, HEADS), generator)
    value, grad = _activations((ROWS, 1, HIDDEN), generator), _activations((ROWS, 1, HIDDEN), generator)
    results = []
    for fused in (False, True):
        a, v, g = _leaves(attention, kv_value, gate)
        if fused:
            out = xsa_head_gate_value(value, a, v, g, HEAD_DIM)
        else:
            out = vllm_value(value, lambda a=a, v=v, g=g: xsa_and_gate_single_rounding(a, v, g, HEAD_DIM))
        out.backward(grad)
        results.append((a.grad, v.grad, g.grad))
    return results


def _query_key(generator, rope: bool):
    raw_query = _activations((ROWS, 1, HEADS, HEAD_DIM), generator)
    raw_key = _activations((ROWS, 1, KV_HEADS, HEAD_DIM), generator)
    query_grad = _activations((ROWS, 1, HEADS, HEAD_DIM), generator)
    key_grad = _activations((ROWS, 1, KV_HEADS, HEAD_DIM), generator)
    inverse = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    angles = torch.outer(torch.arange(ROWS, dtype=torch.float32), inverse)
    freqs = torch.cat((angles, angles), dim=-1).view(ROWS, 1, 1, 64).cuda() if rope else None
    results = []
    for fused in (False, True):
        q, k = _leaves(raw_query, raw_key)
        query_value, key_value = q.detach().clone(), k.detach().clone()
        if fused:
            out_q, out_k = query_key_values(
                query_value, key_value, q, k, (freqs, freqs) if rope else None, QUERY_MULTIPLIER, 1.0
            )
        else:
            ref_q, ref_k = rounded_query_key(
                qk_norm_fp32(q),
                qk_norm_fp32(k),
                (freqs, freqs) if rope else None,
                QUERY_MULTIPLIER,
                1.0,
                torch.bfloat16,
            )
            out_q = vllm_value(query_value, lambda ref_q=ref_q: ref_q)
            out_k = vllm_value(key_value, lambda ref_k=ref_k: ref_k)
        torch.autograd.backward((out_q, out_k), (query_grad, key_grad))
        results.append((q.grad, k.grad))
    return results


@pytest.mark.parametrize(
    "chain",
    [
        _gated_product,
        _swiglu,
        _router,
        _hybrid_input_norm,
        _xsa_head_gate,
        lambda generator: _query_key(generator, rope=True),
        lambda generator: _query_key(generator, rope=False),
    ],
    ids=["gated_product", "swiglu", "router", "hybrid_input_norm", "xsa_head_gate", "query_key_rope", "query_key"],
)
def test_fused_reference_gradients_equal_autograd_of_their_chains_bit_for_bit(chain):
    require_hoppers(1)
    eager, fused = chain(torch.Generator().manual_seed(0))
    _assert_same_bytes(eager, fused)


def test_invariant_router_gives_each_row_the_bytes_of_any_call_holding_it():
    require_hoppers(1)
    generator = torch.Generator().manual_seed(0)
    scale = 10.0 ** torch.empty(300, 1).uniform_(-30, 30, generator=generator)
    x = (torch.randn(300, HIDDEN, generator=generator) * scale).to(torch.bfloat16).cuda()
    weight = (torch.randn(EXPERTS, HIDDEN, generator=generator) * 0.02).to(torch.bfloat16).cuda()
    full = invariant_router_logits(x, weight)

    for rows in (1, 7, 64, 65, 129):
        chunked = torch.cat([invariant_router_logits(chunk, weight) for chunk in x.split(rows)])
        assert torch.equal(chunked.view(torch.int32), full.view(torch.int32))
    for _ in range(20):
        assert torch.equal(invariant_router_logits(x, weight).view(torch.int32), full.view(torch.int32))
    # The engine runs the GEMM on fp32 operands holding the same bf16 values, inside captured CUDA graphs.
    static = x[:65].float().clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        invariant_router_logits(static, weight.float())
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        captured = invariant_router_logits(static, weight.float())
    for start in (65, 130, 0):
        static.copy_(x[start : start + 65].float())
        graph.replay()
        assert torch.equal(captured.view(torch.int32), full[start : start + 65].view(torch.int32))
    exact = x.double() @ weight.double().t()
    bound = 1e-5 * (x.double().abs() @ weight.double().abs().t())
    assert bool(((full.double() - exact).abs() <= bound).all())
