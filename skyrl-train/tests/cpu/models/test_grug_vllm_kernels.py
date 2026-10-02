import pytest
import torch

from skyrl_train.models.grug_vllm_kernels import (
    EXPERT_OFFSET_ELEMENTS,
    expert_weight_offsets,
    serving_engine_ranks,
    serving_row_ranks,
    vllm_ep_combine,
    vllm_qkv_projection,
)


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


def test_ep_combine_adds_rank_partials_in_the_ring_order_of_each_tokens_serving_rank():
    sequences, positions, hidden, num_experts, ep_size, top_k = 4, 16, 32, 16, 4, 4
    tokens = sequences * positions
    generator = torch.Generator().manual_seed(0)
    # Tokens whose four experts sit on one, two, three and four vLLM EP ranks (4 experts per rank).
    selected = torch.stack([torch.randperm(num_experts, generator=generator)[:top_k] for _ in range(tokens)])
    selected[0] = torch.tensor([1, 0, 3, 2])
    selected[1] = torch.tensor([4, 12, 5, 13])
    selected[2] = torch.tensor([9, 2, 8, 15])
    serving = torch.tensor([2, 0, 3, 1])
    # Wide magnitudes so that the order of the bf16 additions changes the result.
    values = torch.randn(tokens, num_experts, hidden, generator=generator) * torch.logspace(-2, 2, num_experts)[:, None]
    # Token 3's slots all sit on rank 0 and cancel in fp32: in slot order (2^24 + 1) - 2^24 is 0, in reverse
    # order (1 - 2^24) + 2^24 is 1, so the rank's partial must follow the slot order.
    selected[3] = torch.tensor([0, 1, 2, 3])
    values[3, :4] = torch.tensor([2.0**24, 1.0, -(2.0**24), 0.0])[:, None]
    values = values.to(torch.bfloat16)
    routing_map = torch.zeros(tokens, num_experts, dtype=torch.bool).scatter(1, selected, True)
    pairs = routing_map.t().nonzero()
    permuted = values[pairs[:, 1], pairs[:, 0]]

    with serving_engine_ranks(serving, data_parallel_size=ep_size, expert_parallel_size=ep_size):
        home, size = serving_row_ranks(tokens)
    combined = vllm_ep_combine(permuted, routing_map, selected, home, size)

    for token in range(tokens):
        slots = values[token, selected[token]]
        # Router row s * B + b holds position s of sequence b.
        home_rank = int(serving[token % sequences])
        expected = _vllm_combine_reference(slots, selected[token].tolist(), home_rank, ep_size, num_experts)
        assert torch.equal(combined[token], expected), token
    # The same slots summed once in fp32 differ, so the test sees the ring's bf16 roundings.
    single = values.gather(1, selected[:, :, None].expand(-1, -1, hidden)).float().sum(1).to(torch.bfloat16)
    assert not torch.equal(combined, single)


def test_ep_combine_of_one_expert_parallel_rank_rounds_once_from_any_serving_rank():
    """Data-parallel engines without expert parallelism (EP 1) add every slot on the serving rank in fp32 and round
    once, so each sequence's serving rank, any rank below the data-parallel size, leaves the sum unchanged."""
    sequences, positions, hidden, num_experts, top_k, data_parallel_size = 2, 8, 16, 8, 4, 2
    tokens = sequences * positions
    generator = torch.Generator().manual_seed(1)
    selected = torch.stack([torch.randperm(num_experts, generator=generator)[:top_k] for _ in range(tokens)])
    values = torch.randn(tokens, num_experts, hidden, generator=generator) * torch.logspace(-2, 2, num_experts)[:, None]
    values = values.to(torch.bfloat16)
    routing_map = torch.zeros(tokens, num_experts, dtype=torch.bool).scatter(1, selected, True)
    pairs = routing_map.t().nonzero()
    permuted = values[pairs[:, 1], pairs[:, 0]]
    once = values.gather(1, selected[:, :, None].expand(-1, -1, hidden)).float().sum(1).to(torch.bfloat16)

    for serving in (torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([-1, 1])):
        with serving_engine_ranks(serving, data_parallel_size=data_parallel_size, expert_parallel_size=1):
            home, size = serving_row_ranks(tokens)
        assert torch.equal(vllm_ep_combine(permuted, routing_map, selected, home, size), once)
    with pytest.raises(ValueError, match="below 2"):
        with serving_engine_ranks(torch.tensor([0, 2]), data_parallel_size=data_parallel_size, expert_parallel_size=1):
            pass


def test_vllm_qkv_projection_has_the_layout_of_megatrons_fused_projection():
    groups, heads_per_group, head_dim, hidden = 3, 2, 4, 8
    generator = torch.Generator().manual_seed(0)
    # Small integers keep every product and sum exact, so a difference can only be a misplaced row.
    fused = torch.randint(-3, 4, (groups * (heads_per_group + 2) * head_dim, hidden), generator=generator).float()
    x = torch.randint(-3, 4, (5, 2, hidden), generator=generator).float()

    projected = vllm_qkv_projection(x, fused, groups, heads_per_group * head_dim, head_dim)

    assert torch.equal(projected, torch.nn.functional.linear(x, fused))


def test_expert_weight_offsets_address_each_expert_from_the_lowest_addressed_weight():
    experts, rows, columns = 4, 3, EXPERT_OFFSET_ELEMENTS
    size = rows * columns
    buffer = torch.arange((experts + 1) * size, dtype=torch.float32)
    # Megatron's parameter buffer holds a layer's experts in reverse order; one bucket boundary leaves a gap.
    starts = [3 * size + size, 2 * size + size, size, 0]
    weights = [buffer[start : start + size].view(rows, columns) for start in starts]
    base = min(weights, key=lambda weight: weight.data_ptr())

    offsets = expert_weight_offsets(weights, base)

    # The kernel reads expert e at ``base + offsets[e] * EXPERT_OFFSET_ELEMENTS`` elements.
    flat = buffer[base.storage_offset() :]
    for weight, offset in zip(weights, offsets, strict=True):
        start = offset * EXPERT_OFFSET_ELEMENTS
        assert torch.equal(flat[start : start + size], weight.flatten())
    misaligned = buffer[1 : 1 + size].view(rows, columns)
    with pytest.raises(ValueError, match="whole"):
        expert_weight_offsets([base, misaligned], base)
