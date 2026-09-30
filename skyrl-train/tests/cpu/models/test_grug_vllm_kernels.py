"""vLLM's expert-parallel combine order and FA3 split counts, as the Grug trainer's numerics reproduce them."""

import pytest
import torch

from skyrl_train.models.grug_vllm_kernels import Fa3Request, fa3_split_counts, vllm_ep_combine


def _vllm_combine_reference(slots, experts, home, ep_size, num_experts):
    """vLLM's combine for one token, written out: per-rank fp32 slot sums, then a bf16 ring from home + 1."""
    per_rank = num_experts // ep_size
    partials = []
    for rank in range(ep_size):
        accumulator = torch.zeros(slots.shape[-1], dtype=torch.float32)
        for slot, expert in enumerate(experts):
            if expert // per_rank == rank:
                accumulator = accumulator + slots[slot].float()
        partials.append(accumulator.to(torch.bfloat16))
    total = partials[(home + 1) % ep_size]
    for step in range(2, ep_size + 1):
        total = (total.float() + partials[(home + step) % ep_size].float()).to(torch.bfloat16)
    return total


def test_ep_combine_adds_rank_partials_in_the_ring_order_of_each_token_home_rank():
    tokens, hidden, num_experts, ep_size, top_k = 64, 32, 16, 4, 4
    generator = torch.Generator().manual_seed(0)
    # Tokens whose four experts sit on one, two, three and four vLLM EP ranks (4 experts per rank).
    selected = torch.stack([torch.randperm(num_experts, generator=generator)[:top_k] for _ in range(tokens)])
    selected[0] = torch.tensor([1, 0, 3, 2])
    selected[1] = torch.tensor([4, 12, 5, 13])
    selected[2] = torch.tensor([9, 2, 8, 15])
    home = torch.randint(0, ep_size, (tokens,), generator=generator)
    # Wide magnitudes so that the order of the bf16 additions changes the result.
    values = torch.randn(tokens, num_experts, hidden, generator=generator) * torch.logspace(-2, 2, num_experts)[:, None]
    values = values.to(torch.bfloat16)
    routing_map = torch.zeros(tokens, num_experts, dtype=torch.bool).scatter(1, selected, True)
    pairs = routing_map.t().nonzero()
    permuted = values[pairs[:, 1], pairs[:, 0]]

    combined = vllm_ep_combine(permuted, routing_map, selected, home, ep_size)

    for token in range(tokens):
        slots = values[token, selected[token]]
        expected = _vllm_combine_reference(slots, selected[token].tolist(), int(home[token]), ep_size, num_experts)
        assert torch.equal(combined[token], expected), token
    # The same slots summed once in fp32 differ, so the test sees the ring's bf16 roundings.
    single = values.gather(1, selected[:, :, None].expand(-1, -1, hidden)).float().sum(1).to(torch.bfloat16)
    assert not torch.equal(combined, single)


@pytest.mark.parametrize(
    ("step", "window", "expected"),
    [
        # One 300-token request, a CUDA-graph step (cap 32): 10 packed query blocks x 3 key blocks, so
        # ceil(30 * 1.1 * 5 / 132) = 2 blocks per SM and ceil(3 / 2) = 2 splits.
        ([Fa3Request(300, 300)], None, [2]),
        # A 700-token request alone (heuristic): 5 KV heads x 22 packed query blocks = 110 >= 0.8 * 132.
        ([Fa3Request(700, 700)], 2048, [1]),
        # A 600-token request alone: 95 query blocks, 5 key blocks; the heuristic allows 4 splits, and
        # ceil(95 * 1.1 * 5 / 132) = 4 blocks per SM gives ceil(5 / 4) = 2.
        ([Fa3Request(600, 600)], None, [2]),
        # A 1,000-token second chunk (300 queries) beside two short prompts in a 400-token step (cap 32):
        # blocks (10 x 8) + (2 x 1) + (2 x 1) = 84, ceil(84 * 1.1 * 5 / 132) = 4 per SM.
        ([Fa3Request(300, 1000), Fa3Request(50, 50), Fa3Request(50, 50)], None, [2, 1, 1]),
        # A 50-token chunk after 1,488 cached tokens, alone (cap 32): 2 x 13 = 26 blocks,
        # ceil(26 * 1.1 * 5 / 132) = 2 per SM, so 7 splits (13 without FA3's 10% margin).
        ([Fa3Request(50, 1538)], None, [7]),
    ],
)
def test_fa3_split_counts_follow_vllm_step_caps_and_the_fa3_heuristic(step, window, expected):
    assert fa3_split_counts(step, kv_heads=5, query_heads_per_kv_head=4, window=window) == expected
