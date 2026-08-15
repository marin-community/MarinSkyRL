# Debugging log for the AIME verifier pilot

Migrate AIME to the shared verifier contracts and restore failed-rollout length observability without changing the
configured reward policy.

## Initial status

`get_rollout_metrics` defines success as reward greater than zero and failure as reward equal to zero. AIME's
default incorrect reward is -1, so incorrect trajectories are omitted from both length buckets. AIME also receives
length and stop reason through a private mutable hook instead of `RolloutEvidence`.

## Hypothesis 1

An explicit `VerificationResult.passed` predicate and its complement can classify both `+1` and `-1` trajectories.

## Changes to make

Add a regression test for a `+1/-1` batch and allow metric projection to consume verifier success predicates.

## Results

The pre-fix `+1/-1` regression failed because `get_rollout_metrics` had no explicit success predicate. After the
change, correct responses average 2.5 tokens and incorrect responses average 10 tokens; negative failures populate
the complementary bucket.

## Hypothesis 2

A native AIME verifier can derive correctness, parseability, and evaluation-budget diagnostics from
`RolloutEvidence`, eliminating the AIME-only generation metadata hook while preserving reward values.

## Changes to make

Add native-verification and aggregation tests, implement the verifier and reward projection, then delete the old
hook and runner calls.

## Results

The pre-fix native-verifier tests failed because AIME returned neither `verification` nor evaluation-budget metrics.
The migrated environment returns both shared contracts, preserves `+1/-1` at zero shaping weight, and reports
conditional correct/incorrect over-budget rates. The parser's `[INVALID]` sentinel is explicitly non-parseable.

## Future work

- [ ] Tune any reward-policy change separately from this contract migration.
