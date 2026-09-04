"""The grouped-MoE combine reduces each token's rows in float32, in ascending expert order.

`combine_routed_rows` replaced a `scatter_add` over repeated token indices. That kernel reduces with
atomics, so the order of the top-k adds changed from launch to launch, and a float32 sum of bf16 rows
is inexact once their exponents span more than 14. On the one-update PPO step the old-log-prob
forward and the training forward then disagreed in the last bit on a few elements, and the routers
downstream amplified that into `log_ratio_abs_max` up to 2.2 (F25, F26, F31). The order is now part
of the contract, so the tests below use rows whose sum depends on it.
"""

from __future__ import annotations

import torch

from skyrl_train.models.layers.moe_routing import TokenReorderer, combine_routed_rows

NUM_EXPERTS = 8
TOP_K = 4
HIDDEN = 3


def _routing(num_tokens: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distinct random experts per token, the production reorderer's row order, and each row's expert."""
    generator = torch.Generator().manual_seed(seed)
    selected = torch.stack([torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K] for _ in range(num_tokens)])
    routing = TokenReorderer(NUM_EXPERTS, TOP_K)(torch.ones(num_tokens, TOP_K), selected)
    expert_of_row = torch.repeat_interleave(torch.arange(NUM_EXPERTS), routing.tokens_per_expert)
    return selected, routing.token_indices, expert_of_row


def test_combine_routed_rows_sums_each_token_in_ascending_expert_order():
    """Rows chosen so the ascending-expert chain gives a closed-form answer that other orders miss.

    Per token t, in ascending expert order, the rows are ``big, 1, -big, 3 + t`` with ``big`` at least
    2**24, so ``big + 1`` rounds back to ``big`` in float32 and the chain reduces to exactly ``3 + t``.
    Four of the 24 orders, the ones that leave the residual last and let the big pair cancel first,
    give the same answer; the other twenty, the reversed chain among them, and a bf16 accumulator do
    not.
    """
    num_tokens = 6
    selected, token_indices, expert_of_row = _routing(num_tokens, seed=3)
    column_scale = torch.arange(1, HIDDEN + 1, dtype=torch.float32)
    rows = torch.empty(num_tokens * TOP_K, HIDDEN)
    for row, (token, expert) in enumerate(zip(token_indices.tolist(), expert_of_row.tolist())):
        rank = sorted(selected[token].tolist()).index(expert)
        big = float(2 ** (24 + token)) * column_scale
        rows[row] = (big, torch.ones(HIDDEN), -big, (3.0 + token) * column_scale)[rank]
    assert torch.equal(rows, rows.to(torch.bfloat16).float()), "fixture values must be exact in bf16"

    combined = combine_routed_rows(rows.to(torch.bfloat16), token_indices, num_tokens, TOP_K)

    expected = (3.0 + torch.arange(num_tokens, dtype=torch.float32)).unsqueeze(1) * column_scale
    assert combined.dtype == torch.float32
    assert torch.equal(combined, expected)


def test_combine_routed_rows_is_exact_when_the_rows_share_a_scale():
    """bf16 rows within a factor of four of each other sum exactly in float32: match an fp64 oracle bitwise."""
    num_tokens = 64
    selected, token_indices, expert_of_row = _routing(num_tokens, seed=11)
    generator = torch.Generator().manual_seed(5)
    magnitude = torch.rand(num_tokens * TOP_K, HIDDEN, generator=generator) * 0.75 + 0.25
    sign = torch.randint(0, 2, magnitude.shape, generator=generator) * 2 - 1
    rows = (magnitude * sign).to(torch.bfloat16)

    combined = combine_routed_rows(rows, token_indices, num_tokens, TOP_K)

    oracle = torch.zeros(num_tokens, HIDDEN, dtype=torch.float64)
    oracle.index_add_(0, token_indices, rows.double())
    assert torch.equal(combined, oracle.float())
