# Debugging log for AIME environment metrics

Restore AIME verifier diagnostics in training metrics without changing rewards or verification.

## Initial status

Snowball training reports every AIME evaluation-budget fraction as zero while the same batch has 57% correct
rollouts. Direct calls to `AIMEEnv.step` and `AIMEEnv.aggregate_metrics` produce the expected values.

## Hypothesis 1

The trajectory runner drops `BaseTextEnvStepOutput.metadata` and sends only `env.get_metrics()` to environment
aggregation. AIME returns verifier and reward diagnostics in step metadata, while its inherited `get_metrics()`
returns an empty dictionary.

## Changes to make

Exercise a correct and an incorrect parseable AIME response through the batched trajectory runner. Require the
aggregated answer-within-budget fraction and default accuracy metric to match those outcomes. Require AIME
aggregation to reject rows that omit its budget diagnostics instead of reporting false values.

## Results

Both regressions fail before the fix. The runner output omits `environment/acc`, and AIME aggregation accepts a
row without budget diagnostics and returns zero fractions.

## Hypothesis 2

Combining the final step metadata with the environment's terminal metrics at the shared runner boundary restores
the diagnostics for both batched and non-batched generation. Direct key access in AIME aggregation exposes any
future plumbing regression.

## Changes to make

Add one shared step-to-metrics projection and use it in whole-trajectory, batched, and step-wise SkyRL-Gym
collection. Replace permissive AIME diagnostic lookups with required-key access.

## Results

The focused regression passes in batched and non-batched modes. A correct parseable response now reports
`environment/acc=1.0` and `environment/answered_within_evaluation_budget_fraction=1.0`. The missing-diagnostic
test raises `KeyError` as required.

The trajectory-runner suite passes 260 tests with one skip. The independent Python 3.12 SkyRL-Gym suite passes
152 tests. The complete launcher and trainer CPU suite passes 1,513 tests with 19 skips.
