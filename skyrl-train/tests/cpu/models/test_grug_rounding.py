"""Grug single-rounding arithmetic against vLLM's formulas evaluated in fp32 and rounded once."""

import pytest
import torch
import torch.nn.functional as F

from skyrl_train.models.grug_moe import GRUG_XSA_EPS
from skyrl_train.models.grug_rounding import (
    gated_norm_product_fp32,
    rotate_neox_fp32,
    swiglu_single_rounding,
    weighted_down_projection_single_rounding,
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


def test_swiglu_rounds_once_where_the_eager_form_rounds_twice():
    fc1 = _bf16(4096, 2 * 8, seed=1)
    gate, up = fc1.float().chunk(2, dim=-1)
    single = (F.silu(gate) * up).to(torch.bfloat16)
    assert torch.equal(swiglu_single_rounding(fc1), single)
    eager = F.silu(fc1[..., :8]) * fc1[..., 8:]
    assert not torch.equal(eager, single)


def test_gated_norm_product_keeps_fp32_until_the_caller_rounds():
    normalized, gate = _bf16(32, 64, seed=2), _bf16(32, 64, seed=3)
    product = gated_norm_product_fp32(normalized, gate)
    assert product.dtype == torch.float32
    assert torch.equal(product, normalized.float() * torch.sigmoid(gate.float()))


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


def test_route_weight_multiplies_the_fp32_down_projection_before_one_rounding():
    splits = [3, 0, 5]
    activation = _bf16(sum(splits), 16, seed=7)
    weights = [_bf16(8, 16, seed=10 + index) for index in range(len(splits))]
    probs = torch.rand(sum(splits), 1, generator=torch.Generator().manual_seed(8)) * 2.5
    rows = activation.float().split(splits)
    fp32 = torch.cat([part @ weight.float().t() for part, weight in zip(rows, weights, strict=True)])
    expected = (fp32 * probs).to(torch.bfloat16)

    result = weighted_down_projection_single_rounding(activation, weights, splits, probs)
    assert torch.equal(result, expected)
    # The trainer's current order weights and rounds the activation before a bf16 projection.
    weighted_first = (activation.float() * probs).to(torch.bfloat16).float().split(splits)
    current = torch.cat(
        [(part @ weight.float().t()).to(torch.bfloat16) for part, weight in zip(weighted_first, weights, strict=True)]
    )
    assert not torch.equal(result, current)
