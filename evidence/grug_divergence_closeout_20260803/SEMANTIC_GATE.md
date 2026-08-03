# Grug eager/grouped semantic gate

Recorded at `2026-08-03T00:33:38Z`, after causal localization and before the
confirmatory eight-H100 run.

## Pinned causal prerequisite

- URI: `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-53d420e/localization-s3.json`
- Source: `53d420eadf0ef2e9cd9d32a8e0efb72ca0ffaabb`
- Payload SHA-256: `976566dc1aa8a882db12326fa84be3d30dbeb2d20565933fbeb7265530e25a3f`
- Canonical result SHA-256: `d890dc2cb8938b6607b4a98a60cfc9476e187695dbc48f6dba599c8a15cb9782`

On all eight ranks, the actual product grouped expert output is bitwise equal
to the manual eager expert output before combine. The layer-zero
selected-expert probe is bitwise equal for output, input gradient, gate-weight
gradient, up-weight gradient, and down-weight gradient. Repeating only the
final BF16 global `index_add_` differs by at most `0.001953125`.

## Confirmatory execution identity

Use one unchanged Ray actor group and rank-to-physical-GPU mapping. Run three
arms in order: eager oracle, eager instrumented, grouped instrumented. Restore
the same checkpoint shards, replay shards, model state, CPU RNG, CUDA RNG, and
empty-gradient baseline before every arm. Do not run an optimizer step.

## Acceptance

The eager instrumentation oracle must pass representative action-log-probability
`rtol=4e-2, atol=4e-3`, global-CE `rtol=2e-3, atol=2e-3`, and representative
gradient `rtol=8e-2, atol=1e-4` checks.

The grouped arm must pass the same action-log-probability and global-CE
tolerances. Every rank must return nonempty finite full-model gradients and
exact route call/allocation accounting at every layer.

For every token, call, and layer, compare the full selected-expert membership.
A membership change is explained only when the eager adjusted-logit margin
between the kth and (k+1)th experts is at most twice that token's maximum
eager/grouped adjusted-logit perturbation. The total unexplained changed-token
count must be zero. Ordered-slot mismatches and exact load-vector differences
remain reported, but are not separate failures when this membership rule
passes.

The pinned selected-expert probe supplies the route-dependent gradient check.
The confirmatory run additionally requires all full-model gradients to be
finite and nonempty. Full-model representative eager/grouped gradient
differences are reported rather than independently gated: an accepted
near-boundary route change makes downstream full-model gradients
discontinuous.

Any identity, oracle, output, CE, route-margin, direct-gradient, or finiteness
failure forbids the 32-H100 pair.
