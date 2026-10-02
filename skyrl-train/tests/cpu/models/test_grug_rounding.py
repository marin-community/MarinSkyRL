import pytest
import torch

from skyrl_train.models.grug_moe import GRUG_XSA_EPS
from skyrl_train.models.grug_rounding import (
    STAGE_STATISTIC_COLUMNS,
    append_stage_statistic,
    rotate_neox_fp32,
    split_stage_statistic,
    vllm_value,
    xsa_and_gate_single_rounding,
)


def _bf16(*shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(torch.bfloat16)


def test_rope_matches_vllm_neox_formula_with_the_bf16_cos_sin_table():
    positions, rotary_dim, head_dim = 64, 8, 16
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    angles = torch.outer(torch.arange(positions, dtype=torch.float), inv_freq)
    # vLLM: cos/sin cache stored in bf16, rotation of the first rotary_dim dims (NeoX halves).
    cos, sin = angles.cos().to(torch.bfloat16).float(), angles.sin().to(torch.bfloat16).float()
    query = _bf16(positions, 1, 2, head_dim, seed=0)
    x = query.float()
    x1, x2 = x[..., : rotary_dim // 2], x[..., rotary_dim // 2 : rotary_dim]
    c, s = cos[:, None, None, :], sin[:, None, None, :]
    expected = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, x[..., rotary_dim:]), dim=-1)

    megatron_freqs = torch.cat((angles, angles), dim=-1)[:, None, None, :]
    assert torch.equal(rotate_neox_fp32(query, megatron_freqs), expected)


@pytest.mark.parametrize("groups", [1, 2])
def test_xsa_and_head_gate_round_once_over_grouped_heads(groups):
    tokens, heads, head_dim = 6, 4, 8
    attention = _bf16(tokens, heads * head_dim, seed=4)
    value = _bf16(tokens, groups, head_dim, seed=5)
    gate = _bf16(tokens, heads, seed=6)
    a = attention.view(tokens, heads, head_dim).float()
    v = value.float().repeat_interleave(heads // groups, dim=1)
    dot = (a * v).sum(-1, keepdim=True)
    projected = a - dot / (v.square().sum(-1, keepdim=True) + GRUG_XSA_EPS) * v
    expected = (projected * (2 * torch.sigmoid(gate.float()))[..., None]).to(torch.bfloat16)

    result = xsa_and_gate_single_rounding(attention, value, gate, head_dim)
    assert torch.equal(result, expected.reshape(tokens, heads * head_dim))


def test_vllm_value_keeps_the_kernels_bytes_and_takes_the_trainers_gradient():
    source = _bf16(8, 4, seed=40).requires_grad_()
    reference = source * 3
    # A kernel value that differs from the reference in its last bits, and holds -0.0, which ``x + 0`` would turn to +0.
    kernel = (reference.detach().float() * (1 + 2**-9)).to(torch.bfloat16)
    kernel[0, 0] = -0.0
    value = vllm_value(kernel, lambda: reference)
    assert torch.equal(value.view(torch.int16), kernel.view(torch.int16))
    value.backward(torch.ones_like(value))
    assert torch.equal(source.grad, torch.full_like(source, 3))
    with torch.no_grad():
        assert vllm_value(kernel, lambda: reference) is kernel


def test_stage_statistic_crosses_a_pipeline_boundary_bit_for_bit_and_only_the_hidden_states_take_gradient():
    sequence, batch, hidden_size = 5, 2, 8
    hidden = _bf16(sequence, batch, hidden_size, seed=0).requires_grad_()
    # Statistics whose bf16 halves are NaN, infinity and subnormal patterns, and signed zeros: any arithmetic or bf16
    # conversion on the way changes their bits.
    statistic = torch.tensor([float("nan"), -0.0, 0.0, 3.0e-39, -1.0e38, 1.0 + 2.0**-23, float("inf"), 1.0e-45])
    statistic = statistic.repeat(2)[: sequence * batch].view(sequence, batch, 1)
    statistic.view(torch.int32)[0, 0, 0] = 0x7F80FFFF  # a NaN whose low bf16 word is a NaN pattern as well

    packed = append_stage_statistic(hidden, statistic)
    assert packed.dtype == torch.bfloat16 and packed.shape == (sequence, batch, hidden_size + STAGE_STATISTIC_COLUMNS)
    # The next stage receives the packed tensor as a leaf requiring gradient, as Megatron's pipeline delivers it.
    received = packed.detach().clone().requires_grad_()
    received_hidden, received_statistic = split_stage_statistic(received, hidden_size)
    assert torch.equal(received_hidden.view(torch.int16), hidden.detach().view(torch.int16))
    assert torch.equal(received_statistic.view(torch.int32), statistic.view(torch.int32))

    weight = _bf16(sequence, batch, hidden_size, seed=1)
    (received_hidden.float() * weight.float()).sum().backward()
    assert torch.equal(received.grad[..., :hidden_size], weight)
    assert not received.grad[..., hidden_size:].any()
    # The previous stage backpropagates the received gradient into its hidden states alone.
    packed.backward(received.grad)
    assert torch.equal(hidden.grad, weight)
