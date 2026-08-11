# Debugging log for trajectory reward shaping

Make loop, non-termination, and successful-response length penalties independent of the trajectory generator and preserve raw task outcomes for success metrics.

## Initial status

SkyRL Gym and StepWise can zero non-stopping rewards or mask overlong samples. TerminalBench has separate termination and repeated-action shaping. Every generator already returns through `GeneratorInterface.generate()`, but that shared boundary only finalizes alignment metrics.

## Hypothesis 1

A post-generation stage at `GeneratorInterface.generate()` can shape every normalized `GeneratorOutput` without coupling behavior to SkyRL Gym, StepWise, TerminalBench, Verifiers, or an inference backend.

## Changes to make

Define one typed trajectory-shaping configuration and one pure batch transformation. Preserve the raw task outcome in `unshaped_rewards`, apply additive penalties to the optimization reward, and emit component metrics and detected loop spans. Keep the default disabled.

## Results

The shared boundary shapes scalar and token-level rewards, preserves the raw outcome channel, and records component values, loop spans, stop-reason counts, and schema version. A rolling token-window detector finds repeated spans in linear time and verifies hash matches against the original tokens. It resets at non-trainable regions so tool observations separate assistant turns.

## Hypothesis 2

The correct-response length preference can use the same additive component contract while remaining success-metric invariant if eligibility is based only on a positive raw outcome.

## Changes to make

Add a bounded linear penalty over trainable response tokens beyond a configurable free-token allowance. Assert that longer correct responses receive less optimization reward while zero-reward responses and pass-rate metrics are unchanged.

## Results

The positive-outcome gate leaves zero-reward responses unchanged. A two-token correct response kept reward 1.0 while a five-token correct response received 0.7 under a 0.1-per-token penalty with two free tokens; pass@2 remained based on the raw outcome. Step-wise output accumulates response length across rows and applies the penalty to the final row consumed by the estimator.

## Validation

- The first red contract failed at import because no shared shaping module existed.
- Focused shaping, generator, configuration, and trainer-utility tests pass, including async concatenation and dynamic sampling.
- The repository's YAML-policy test caught an explanatory comment in the root config; the comment was removed and the contract now passes.
- The full launcher and trainer CPU gate passes: 1,316 passed and 21 skipped.

## Future work

- [ ] Validate the shaping coefficients in a matched Hero-v2 continuation.
