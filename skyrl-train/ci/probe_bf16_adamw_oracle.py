"""F21 -- does the SHIPPED BFloat16AdamW track a changing gradient, measured against fp64?

A script, not a test. The behaviour it measures is currently WRONG, so a passing test asserting it
would go red the day someone fixes the optimizer -- which is why the earlier characterisation suite
was deleted rather than kept (gate G2 pass 9). Run it, read the numbers, put them in the ledger.

What both gates asked for, and what the deleted suite did NOT do:
  - drive the REAL optimizer, not a reimplementation of the recurrence;
  - start `exp_avg_sq` from ZERO, as `bf16_adamw.py` does, not from the equilibrium v = g^2 = 1;
  - compare against an FP64 reference, not fp32;
  - move the gradient BOTH ways, up and down.

Run:
  cd skyrl-train && uv run --frozen python ci/probe_bf16_adamw_oracle.py
"""

from __future__ import annotations

import torch

from skyrl_train.distributed.bf16_adamw import BFloat16AdamW, BFloat16UpdateMode

LR = 1e-6
BETAS = (0.9, 0.999)
EPS = 1e-8


def _reference(grads: list[float]) -> list[float]:
    """exp_avg_sq under the same recurrence in fp64, from zero. What Adam intends."""
    v = torch.zeros(1, dtype=torch.float64)
    out = []
    for g in grads:
        gt = torch.tensor([g], dtype=torch.float64)
        v = v * BETAS[1] + (1 - BETAS[1]) * gt * gt
        out.append(float(v.item()))
    return out


def _shipped(grads: list[float]) -> list[float]:
    """exp_avg_sq as the SHIPPED optimizer actually evolves it."""
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    # stochastic is E6's configured mode; the second moment is mode-independent, but drive the
    # shipped configuration rather than a convenient one.
    optimizer = BFloat16AdamW(
        [parameter], update_mode=BFloat16UpdateMode.STOCHASTIC, seed=17, lr=LR, betas=BETAS, eps=EPS
    )
    out = []
    for g in grads:
        parameter.grad = torch.tensor([g], dtype=torch.bfloat16)
        optimizer.step()
        out.append(float(optimizer.state[parameter]["exp_avg_sq"].float().item()))
    return out


def main() -> None:
    # Up, then down: a schedule that a working second moment must follow in both directions.
    schedule = [1.0] * 3000 + [3.0] * 3000 + [0.5] * 3000
    shipped = _shipped(schedule)
    reference = _reference(schedule)

    print(f"{'step':>6}  {'g':>6}  {'shipped v':>12}  {'fp64 v':>12}  {'abs err':>12}  {'rel err':>9}")
    for i in (0, 100, 1000, 2999, 3100, 4000, 5999, 6100, 7000, 8999):
        err = abs(shipped[i] - reference[i])
        rel = err / max(reference[i], 1e-30)
        print(f"{i:>6}  {schedule[i]:>6.2f}  {shipped[i]:>12.6f}  {reference[i]:>12.6f}  {err:>12.6f}  {rel:>8.1%}")

    worst = max(abs(s - r) / max(r, 1e-30) for s, r in zip(shipped, reference))
    print()
    print(f"worst relative error in exp_avg_sq over {len(schedule)} steps: {worst:.1%}")
    print(f"final: shipped {shipped[-1]:.6f} vs fp64 {reference[-1]:.6f}")
    moved_up = shipped[5999] > shipped[2999]
    moved_down = shipped[-1] < shipped[5999]
    print(f"tracked the gradient UP:   {moved_up}")
    print(f"tracked the gradient DOWN: {moved_down}")

    # The direction claim on its own, with the gradient driven to EXACTLY zero. `v` decays only
    # through the `v *= beta2` multiply, which is below half an ulp in bf16, so it never runs.
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    optimizer = BFloat16AdamW(
        [parameter], update_mode=BFloat16UpdateMode.STOCHASTIC, seed=17, lr=LR, betas=BETAS, eps=EPS
    )
    for g in [4.0] * 4000:
        parameter.grad = torch.tensor([g], dtype=torch.bfloat16)
        optimizer.step()
    peak = float(optimizer.state[parameter]["exp_avg_sq"].float().item())
    for _ in range(20000):
        parameter.grad = torch.zeros(1, dtype=torch.bfloat16)
        optimizer.step()
    after = float(optimizer.state[parameter]["exp_avg_sq"].float().item())
    print()
    print(f"after 4,000 steps at g=4:            v = {peak}")
    print(f"after 20,000 further steps at g=0:   v = {after}")
    print(f"decayed at all? {after < peak}    (fp64 would reach {peak * BETAS[1] ** 20000:.3e})")


if __name__ == "__main__":
    main()
