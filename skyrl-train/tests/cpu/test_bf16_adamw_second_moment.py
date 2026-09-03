# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does BFloat16AdamW's second moment actually track the gradient?

`_second_moment_update` allocates `exp_avg_sq` with `zeros_like(parameter_local)`, so on a bf16
parameter the second moment is bf16 too, and the update is `v <- v * beta2 + (1 - beta2) * g^2`.
At the configured beta2 = 0.999 the decay multiply is below half an ulp of bf16 across the whole
normal range, so `v * beta2` rounds back to `v` and the increment is absorbed unless `g^2` differs
from `v` by enough to reach the next representable value.

These tests are CPU-only and exact. They characterise the CURRENT behaviour rather than assert a
fix, so the number is on the record before any GPU time is spent on it: an optimizer whose variance
estimate is frozen is invisible to the PPO ratio invariant, which is evaluated BEFORE the optimizer
step and so cannot see the optimizer evolve wrongly.
"""

from __future__ import annotations

import torch


BETA2 = 0.999


def _second_moment_step(v: torch.Tensor, grad: torch.Tensor, beta2: float = BETA2) -> torch.Tensor:
    """One `exp_avg_sq` update, exactly as bf16_adamw performs it."""
    out = v.clone()
    out.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    return out


def test_the_bf16_decay_multiply_is_a_no_op_at_beta2_0_999():
    """`v * 0.999` rounds back to `v` in bf16. The decay half of the update does nothing.

    bf16 carries 8 significand bits, so the gap between neighbours is 2^-8 relative — a 3.9e-3 step
    against the 1e-3 that multiplying by 0.999 asks for. The product is always nearer `v` than the
    next value down.
    """
    for value in (1.0, 0.5, 3.7, 1e-3, 1e3):
        v = torch.tensor([value], dtype=torch.bfloat16)
        assert (v * BETA2).item() == v.item(), f"decay moved v={value}, which bf16 cannot represent"


def test_the_second_moment_is_frozen_until_the_gradient_squared_deviates_several_fold():
    """A steady gradient never moves the variance estimate, and a modest change does not either.

    This is the shape of the defect: `v` is stuck at its first value while `g^2` drifts, so the
    per-parameter step size stops adapting.
    """
    v = torch.tensor([1.0], dtype=torch.bfloat16)
    steady = torch.tensor([1.0], dtype=torch.bfloat16)
    for _ in range(1000):
        v = _second_moment_step(v, steady)
    assert v.item() == 1.0, "1,000 steps of a steady gradient left the estimate unchanged"

    for ratio in (1.5, 2.0, 3.0):
        moved = _second_moment_step(torch.tensor([1.0], dtype=torch.bfloat16), torch.tensor([ratio**0.5]).bfloat16())
        assert moved.item() == 1.0, f"g^2 = {ratio}x the estimate still did not move it"

    # It only starts tracking once the increment clears half an ulp on its own.
    far = _second_moment_step(torch.tensor([1.0], dtype=torch.bfloat16), torch.tensor([2.0], dtype=torch.bfloat16))
    assert far.item() > 1.0, "g^2 = 4x must move it, or the characterisation above is wrong"


def test_an_fp32_second_moment_tracks_the_same_gradients():
    """The control. Identical arithmetic in fp32 converges toward g^2, which is what Adam intends.

    Run beside the bf16 case, this is what says the freeze is a dtype artifact rather than the
    algorithm.
    """
    # beta2 = 0.999 gives a time constant of 1/(1 - beta2) = 1,000 steps, so 5,000 steps closes
    # 1 - 0.999^5000 = 99.3% of the gap. Asserting convergence in fewer would be asserting the wrong
    # thing about the algorithm rather than about the dtype.
    steps = 5000
    target = 1.5
    grad = torch.tensor([target**0.5], dtype=torch.float32)

    v32 = torch.tensor([1.0], dtype=torch.float32)
    for _ in range(steps):
        v32 = _second_moment_step(v32, grad)
    closed = (v32.item() - 1.0) / (target - 1.0)
    assert closed > 0.99, f"fp32 should close ~99% of the gap in {steps} steps, closed {closed:.3f}"

    v16 = torch.tensor([1.0], dtype=torch.bfloat16)
    grad16 = grad.bfloat16()
    for _ in range(steps):
        v16 = _second_moment_step(v16, grad16)
    assert v16.item() == 1.0, "bf16 closed NONE of it, over the same 5,000 steps"


def test_the_resident_cost_of_an_fp32_second_moment_is_two_bytes_per_parameter_per_shard():
    """The number the fix would be judged against, stated exactly rather than approximated.

    Optimizer state is sharded across the POLICY ranks only; the inference GPUs hold none. Getting
    the divisor or the binary/decimal prefix wrong is how this was first published as 1.68 GiB.
    """
    params = 67_078_876_160
    delta = params * (4 - 2)
    per_smoke_rank = delta / 16 / 2**30
    per_e6_rank = delta / 64 / 2**30
    assert round(per_smoke_rank, 3) == 7.809
    assert round(per_e6_rank, 3) == 1.952
