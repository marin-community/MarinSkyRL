# Debugging log for train CI after PR 369

Restore the CPU trainer gate after PRs 368 and 369 added default configuration fields.

## Hypothesis 1

The failures are config-fixture drift and an unintended hardware probe, not failures in colocated engine placement.

## Changes to make

Reproduce the structural snapshot, partial worker config, and batch-invariant validation failures. Keep the peer-
access boundary fake in tests that only exercise environment propagation.

## Results

CI reports six failures: one stale additive-config snapshot, two partial worker configs missing `batch_invariant`,
two tests invoking Ray's real GPU peer-access probe on a CPU runner, and one generic SGLang validation that runs
before the batch-invariant compatibility check. The eight directly affected tests pass after updating those
boundaries.

## Future work

- [ ] None.
