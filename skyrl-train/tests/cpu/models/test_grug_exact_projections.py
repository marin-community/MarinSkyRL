import pytest
import torch

from skyrl_train.models.grug_exact.context import ExactForward
from skyrl_train.models.grug_exact.projections import qkv_projection, split_qkv
from skyrl_train.models.grug_exact.vllm_kernels import EXPERT_OFFSET_ELEMENTS, expert_weight_offsets


def test_qkv_projection_has_the_layout_of_megatrons_fused_projection():
    groups, heads_per_group, head_dim, hidden = 3, 2, 4, 8
    generator = torch.Generator().manual_seed(0)
    # Small integers keep every product and sum exact, so a difference can only be a misplaced row.
    fused = torch.randint(-3, 4, (groups * (heads_per_group + 2) * head_dim, hidden), generator=generator).float()
    x = torch.randint(-3, 4, (5, 2, hidden), generator=generator).float()

    projected = qkv_projection(x, fused, groups, heads_per_group * head_dim, head_dim)

    assert torch.equal(projected, torch.nn.functional.linear(x, fused))
    query, key, value = split_qkv(projected, groups, heads_per_group * head_dim, head_dim)
    grouped = projected.view(5, 2, groups, (heads_per_group + 2) * head_dim)
    assert torch.equal(query.reshape(5, 2, groups, -1), grouped[..., : heads_per_group * head_dim])
    assert torch.equal(key, grouped[..., heads_per_group * head_dim : (heads_per_group + 1) * head_dim])
    assert torch.equal(value, grouped[..., (heads_per_group + 1) * head_dim :])


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


def test_exact_forward_gives_each_router_row_its_sequences_serving_rank_and_hands_attention_to_one_recompute():
    forward = ExactForward(serving_ranks=torch.tensor([2, 0, 3]), expert_parallel_size=4, keep_attention=True)
    # Router row s * B + b holds position s of sequence b.
    assert forward.row_serving_ranks(6).tolist() == [2, 0, 3, 2, 0, 3]
    with pytest.raises(ValueError, match="evenly"):
        forward.row_serving_ranks(7)

    attention = torch.ones(2)
    forward.keep(3, attention)
    with pytest.raises(RuntimeError, match="layers \\[3\\]"):
        forward.assert_drained()
    assert forward.kept_attention(3) is attention
    assert forward.kept_attention(3) is None
    forward.assert_drained()
    # A forward without a recompute keeps nothing.
    plain = ExactForward(serving_ranks=torch.tensor([0]), expert_parallel_size=1, keep_attention=False)
    plain.keep(1, attention)
    assert plain.kept_attention(1) is None
