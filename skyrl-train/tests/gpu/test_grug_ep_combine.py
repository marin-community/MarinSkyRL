"""H100 check that the ``ep_sum`` combine's Triton kernel adds each token's slots as the CPU combine does."""

import pytest
import torch

from skyrl_train.models.grug_vllm_kernels import vllm_ep_combine
from tests.gpu.grug_gpu_gates import require_hoppers

HIDDEN = 2560
EXPERTS = 256
TOP_K = 4
EP_SIZE = 8
EXPERTS_PER_RANK = EXPERTS // EP_SIZE


def _combine_inputs(tokens: int, generator: torch.Generator):
    selected = torch.rand(tokens, EXPERTS, generator=generator).argsort(dim=1)[:, :TOP_K]
    # Token 0's slots share rank 0 and cancel in fp32 unless added in slot order; token 1's slots sit on four ranks.
    selected[0] = torch.tensor([3, 0, 2, 1])
    if tokens > 1:
        selected[1] = torch.arange(TOP_K) * EXPERTS_PER_RANK + 5
    home = torch.randint(0, EP_SIZE, (tokens,), generator=generator)
    routing_map = torch.zeros(tokens, EXPERTS, dtype=torch.bool).scatter(1, selected, True)
    # Magnitudes from 1e-6 to 1e6 make both the fp32 rank sums and the bf16 ring additions round.
    magnitudes = 10.0 ** torch.empty(tokens * TOP_K, 1).uniform_(-6.0, 6.0, generator=generator)
    permuted = torch.randn(tokens * TOP_K, HIDDEN, generator=generator) * magnitudes
    permuted[:, :8] = 0.0
    permuted[:, 8:16] = -0.0
    permuted[:, 16:24] = torch.randn(tokens * TOP_K, 8, generator=generator) * 1e-39
    rows = routing_map.t().nonzero()
    token_zero = (rows[:, 1] == 0).nonzero().flatten()
    # Megatron's permuted rows run by expert, so token 0's rows hold experts 0, 1, 2, 3 in turn.
    permuted[token_zero, 24:] = torch.tensor([2.0**24, -(2.0**24), 1.0, 0.0])[:, None]
    return permuted.to(torch.bfloat16), routing_map, selected, home


@pytest.mark.parametrize("tokens", [1, 7, 913, 4096])
def test_ep_combine_on_cuda_matches_the_cpu_combine_bit_for_bit(tokens):
    require_hoppers(1)
    generator = torch.Generator().manual_seed(tokens)
    permuted, routing_map, selected, home = _combine_inputs(tokens, generator)

    expected = vllm_ep_combine(permuted, routing_map, selected, home, EP_SIZE)
    combined = vllm_ep_combine(permuted.cuda(), routing_map.cuda(), selected.cuda(), home.cuda(), EP_SIZE)

    assert torch.equal(combined.cpu().view(torch.int16), expected.view(torch.int16))
    # Token 0 in slot order: (2^24 + 1) - 2^24 rounds to 0 in fp32, where expert order would give 1.
    assert torch.equal(expected[0, 24:], torch.zeros(HIDDEN - 24, dtype=torch.bfloat16))
