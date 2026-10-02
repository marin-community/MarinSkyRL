import torch

from skyrl_train.models.grug_invariant_kernels import invariant_router_logits
from tests.gpu.grug_gpu_gates import require_hoppers

# Grug's router: 2,560 hidden columns, 256 experts.
HIDDEN, EXPERTS = 2560, 256


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
