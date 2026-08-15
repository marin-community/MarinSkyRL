# AIME verifier pilot

## Objective

Migrate AIME from an environment-local scalar scorer and an AIME-specific generation-metadata hook to the shared
verification contracts. The migration must preserve configured optimization rewards while making correctness,
answer completion, and response length observable from the same rollout evidence used to verify the answer.

## Current failure

The generic rollout metrics infer success from a positive reward and failure from a zero reward. AIME's default
reward scale is `+1/-1`, so incorrect trajectories enter neither bucket and
`generate/avg_tokens_zero_rewards` remains zero. Separately, AIME receives response length and stop reason through
`set_generation_metadata`; this duplicates the runner-to-verifier boundary and does not express the evaluation
budget or whether a parseable answer was produced within it.

## Native AIME boundary

`AIMEVerifier` implements the shared verifier protocol:

```text
RolloutEvidence(response, stop_reason, generated_token_count, token ids)
    -> AIMEVerifier(ground_truth, evaluation_token_budget)
    -> VerificationResult(score=+1/-1, passed=correct, diagnostics=...)
    -> AIME reward policy
    -> RewardResult(unshaped_reward, optimization_reward, components)
    -> TrainingDisposition
```

The verifier's score is the unshaped `+1/-1` correctness verdict. Diagnostics carry the parsed prediction,
generated-token count, whether the response exceeded the evaluation budget, and whether a parseable answer was
produced within that budget. Verification never applies optimization policy.

The AIME reward policy preserves the existing `length_penalty_weight`, `target_length`, `truncated_penalty`, and
`min_response_length` behavior. It consumes the verifier result and rollout evidence rather than mutable fields on
the environment. The policy returns a `RewardResult`; the environment projects its optimization scalar onto the
legacy `BaseTextEnvStepOutput.reward` field while also returning the native verification result.

## Metrics

Trajectory projections pass each verifier's explicit `passed` value to `get_rollout_metrics`. When a verifier does
not define `passed`, the existing positive-reward success predicate remains the fallback. The complementary failure
mask is always the inverse of the success mask, so negative rewards cannot disappear from both token-length buckets.
The existing metric names remain stable for dashboards.

AIME environment aggregation reports:

- the fraction of all responses over the evaluation token budget;
- the conditional fraction of correct responses over budget;
- the conditional fraction of incorrect responses over budget;
- the fraction producing a parseable answer within budget.

The default evaluation token budget is 8,192 and is configurable under `environment.skyrl_gym.aime`.

## Removed mechanism

`AIMEEnv.set_generation_metadata` and both runner call sites are deleted. `BaseTextEnv.set_rollout_evidence` is the
only runner-to-environment publication hook. No second length or termination mitigation is introduced, and the
existing reward policy remains authoritative.

## Compatibility and tests

- With `length_penalty_weight=0`, correct and incorrect AIME rewards remain exactly `+1/-1`.
- Existing non-AIME environments continue to receive legacy scalar-to-verification adaptation.
- A `+1/-1` metric regression test proves both success and failure token buckets populate.
- AIME tests prove that verification consumes token evidence, reports over-budget failures, and distinguishes a
  missing/unparseable answer from a correct answer.
- Runner tests prove evidence reaches AIME before `step` without the removed metadata hook.
