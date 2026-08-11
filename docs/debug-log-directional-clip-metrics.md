# Debugging log for directional clip metrics

Separate lower- and upper-bound policy clipping while preserving the pooled metric used by existing dashboards.

## Initial status

Every policy loss returns one scalar clipping fraction. PPO and GSPO pool objective clipping from both ratio bounds,
CISPO pools both clamped-ratio conditions, and Clip-Cov reports its covariance correction fraction through the same
field. A sweep of `eps_clip_high` therefore cannot tell whether the upper bound fired. The policy-loss and registry
code also lives in the 1,815-line `ppo_utils.py` module.

## Hypothesis 1

The policy-loss result needs structured diagnostics. Computing lower and upper pressure from the ratio, then
intersecting each side with the loss-specific clipping decision, distinguishes ratio-side behavior without inferring
the side from advantage sign.

## Changes to make

Add numerical regression cases with mixed ratio sides and advantages. The cases will vary one epsilon at a time and
assert that only its matching ratio side changes. They will also require separate lower, upper, pressure, and pooled
metrics.

## Results

The numerical side-isolation test passed before the fix: `eps_clip_low` already controls only ratios below one and
`eps_clip_high` only ratios above one. The diagnostics test failed because the policy-loss result exposed only the
pooled `0.5` fraction.

The training-backend contract now completes the metric dictionary, while each policy loss reports only the metrics
it owns. PPO reports a `0.25` lower clipped fraction, a `0.25` upper clipped fraction, and matching pressure values
for the four-token regression batch while retaining the pooled `0.5` value. The ratio side is determined from the
unclamped ratio and intersected with the objective's actual clipping decision; it is not inferred from the advantage
sign. CISPO and GSPO use the same contract. Losses without ratio clipping return no clipping metrics; the FSDP2 and
Megatron boundaries fill the complete neutral key set. Clip-Cov preserves its historical pooled covariance fraction
while reporting the standard PPO bound decisions separately.

The former module was split into registry machinery, an algorithm registry, shared policy math, KL controllers,
importance-ratio diagnostics, policy losses, loss reduction, and advantage estimators. Imports and custom-loss
examples now use those named modules without a compatibility shim. The stale pre-accumulator importance-ratio
implementation was deleted.

Both FSDP2 and Megatron forward the complete directional metric key set. Package initialization registers the full
built-in algorithm set, including `rloo_n_pbs`; config validation and Ray actor restarts preserve local user
registrations. The focused policy-loss and backend suites pass with 60 tests. The complete launcher and trainer CPU
gate passes with 1,300 tests and 20 skips.

## Future work

- [ ] Record whether the split metrics make the next asymmetric clipping sweep interpretable in practice.
