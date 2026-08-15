# Shared verifier contracts

## Scope

Shared verifier contracts govern the Harbor and SkyRL-Gym runner boundaries without defining an environment's
scoring policy. Environment migrations, including AIME's scoring and outcome-length behavior, build on this seam
but retain their own designs and regression coverage.

## Problem

Trajectory runners currently receive verifier information in incompatible shapes:

- Harbor returns messages and `rollout_details` through `TrialResult.agent_result`, a nullable
  `TrialResult.verifier_result`, and exception metadata. `HarborTrajectoryRunner` interprets all of those fields,
  shapes rewards, and independently decides token masking and RLOO-N baseline membership.
- SkyRL-Gym returns a `BaseTextEnvStepOutput` whose scalar `reward` may be a verifier verdict, an already-shaped
  optimization reward, or both. Verifier diagnostics are untyped entries in `metadata`. The runner cannot tell a
  failed verification from an unavailable verifier or recover the raw outcome after an environment shapes it.

The ambiguity permits invalid states. A missing verifier can look like a score of zero; a shaped reward can be
reported as verifier accuracy; token credit can disagree with the scalar reward; and error handling can mask a
trajectory without consistently excluding it from a baseline. It also forces each harness integration to invent
its own seam between rollout collection, verification, reward policy, and training eligibility.

## Goals

1. Give every trajectory runner the same four typed values:

   | Existing harness data | Shared contract |
   | --- | --- |
   | conversation, generated response, token/logprob evidence | `RolloutEvidence` |
   | verifier verdict, diagnostics, or verifier unavailability | `VerificationResult` |
   | raw outcome, optimization total, token rewards, components, token credit | `RewardResult` |
   | loss eligibility, baseline eligibility, and failure reason | `TrainingDisposition` |

2. Make unavailable verification distinct from a valid zero or negative score.
3. Keep verifier logic independent of Harbor, SkyRL-Gym, inference engines, and trainer batch dictionaries.
4. Preserve existing rewards, retry behavior, masks, and metrics while migrating the runner boundaries.
5. Establish a native verifier protocol that an environment such as AIME can adopt without moving numerical
   verification into a trajectory runner.

## Non-goals

- Rewriting all SkyRL-Gym environments as part of the contract migration.
- Changing AIME parsing, reward values, length shaping, or evaluation budgets. Those changes belong to the AIME
  AIME migration.
- Standardizing harness execution, sandbox lifecycle, or retry policy. These remain trajectory-runner concerns.
- Putting inference-engine calls inside verifiers. A verifier consumes immutable rollout evidence; it does not
  generate or mutate a trajectory.
- Replacing `TrajectoryBatch`, which remains the trainer-facing batch transport.

## Ownership and package boundary

The contracts live in `skyrl_gym.verification`. `skyrl-gym` is independently testable and cannot import
`skyrl_train`; the root trainer already depends on `skyrl-gym`. Placing the lowest-level contracts in the gym
package lets both maintained in-process verifiers and trainer-side Harbor adapters use the same definitions
without creating a dependency cycle.

The module has no dependency on Hydra, Ray, Harbor, tokenizers, or torch. Contract tests run in the independent
`skyrl-gym` test suite.

## Contracts

### `RolloutEvidence`

`RolloutEvidence` is an immutable semantic record of what happened, not a dump of a harness object. It carries:

- normalized conversation messages and the final response text;
- stop reason and generated-token count;
- optional prompt/response token IDs and behavior logprobs when the runner has them;
- immutable diagnostic metadata for verifier-specific evidence that has no common field.

Harness adapters must translate their native records at the boundary. Harbor's `rollout_details` are therefore
decoded into token IDs and logprobs before constructing evidence; the raw Harbor object does not leak into the
contract. SkyRL-Gym constructs evidence from the model response and generation metadata before calling `step`.

### `VerificationResult`

`VerificationResult` has an explicit status:

- `VERIFIED`: the verifier ran and produced a numeric score;
- `UNAVAILABLE`: no verdict exists, with a machine-readable reason;
- `ERROR`: the verifier attempted verification but failed, with a machine-readable reason.

Only `VERIFIED` may carry a score. A score of `0.0` or `-1.0` is a real verdict, never a missing-result sentinel.
Optional `passed` and diagnostics are verifier facts; they do not encode training treatment.

### `RewardResult`

`RewardResult` separates the raw verifier outcome from the reward optimized by the trainer. It carries the
unshaped scalar outcome, a scalar optimization total, optional per-token reward placement, named shaping
components, and optional token-level credit. Shared construction validates finite numerics and token-channel
lengths when evidence contains response token IDs.

The raw outcome is absent when verification is unavailable. A zero optimization reward may still be assigned by
error policy, but it cannot masquerade as a verifier outcome.

### `TrainingDisposition`

`TrainingDisposition` represents two independent decisions:

- whether generated tokens are eligible for loss;
- whether the sample participates in a group baseline.

It also carries a stable reason and optional exception type. This directly represents the current Harbor cases:

| Current treatment | loss eligible | baseline eligible |
| --- | ---: | ---: |
| valid or pass-through with verifier evidence | yes | yes |
| preserved zero-reward generation | yes | policy-dependent |
| agent failure stub | no | yes |
| infrastructure/missing-verifier failure | no | no |
| pass-through lacking required behavior logprobs | no | no |

Masking is derived from the disposition when projecting into `TrajectoryBatch`; callers do not separately mutate
`loss_mask` and `exclude_from_baseline`.

## Protocols and data flow

The native synchronous protocol is intentionally small:

```python
class Verifier(Protocol):
    def verify(self, evidence: RolloutEvidence) -> VerificationResult: ...
```

Verifiers are pure scoring components. Task-specific ground truth or tools are constructor dependencies. They do
not shape rewards, decide retries, mask loss, or know about a trainer batch. Async or remote harnesses may execute
verification elsewhere; their runner adapter must still return the same `VerificationResult`.

Reward and disposition policies consume the contracts after verification:

```text
model or external harness
          |
          v
  RolloutEvidence ----> Verifier or harness adapter ----> VerificationResult
          |                                                   |
          +-------------------> reward policy ----------------+----> RewardResult
          |                                                   |
          +----------------> disposition policy --------------+----> TrainingDisposition
                                                                      |
                                                                      v
                                                             TrajectoryBatch projection
```

This is a data contract, not a requirement that every verifier execute in the runner process.

## Migration

### SkyRL-Gym

`BaseTextEnvStepOutput` gains an optional `verification` field. Existing environments remain source-compatible.
The SkyRL-Gym trajectory runner normalizes each step as follows:

- a native `VerificationResult` is preserved;
- a legacy scalar `reward` is adapted to `VERIFIED` with the same score;
- raw and optimization rewards remain identical for legacy environments;
- the normal disposition is loss-eligible and baseline-eligible.

The runner's internal interaction result stores the four contracts. Its final projection derives the existing
`TrajectoryBatch` fields, preserving current trainer behavior. The AIME migration replaces its legacy scalar
adaptation with a native verifier and explicit reward policy.

### Harbor

A Harbor adapter converts `TrialResult` into evidence and verification before reward shaping and training
treatment. Missing `verifier_result` becomes `UNAVAILABLE`; verifier exceptions become `ERROR`; numeric verifier
scores, including zero, become `VERIFIED`.

The existing error-classification configuration remains authoritative. The adapter records its decision in a
`TrainingDisposition`, and records the output of existing Harbor reward shapers in a `RewardResult`.
`TerminalBenchAgentOutput` stores those contracts without parallel compatibility fields.

### Trainer batch

The runner projection is the only place that maps contracts onto legacy batch keys:

- `RewardResult.optimization_reward` or its explicit token placement -> `rewards`;
- `RewardResult.unshaped_reward` -> `unshaped_rewards`;
- reward components remain available to runner metrics; token credit -> its existing optional channel;
- `TrainingDisposition.loss_eligible` -> loss-mask preservation or zeroing;
- `not TrainingDisposition.baseline_eligible` -> `exclude_from_baseline`.

The trainer does not depend on verifier classes.

## Invariants and tests

Contract tests will reject:

- `VERIFIED` without a finite score;
- `UNAVAILABLE` or `ERROR` with a score;
- non-finite raw, shaped, component, or token-credit values;
- token rewards or token credit whose length disagrees with available response-token evidence;
- a baseline-eligible disposition without an explicit reason when loss is ineligible.

Runner regression tests will cover:

1. Harbor score `0.0` remains verified rather than unavailable.
2. Harbor missing verifier results retain their current mask and baseline treatment.
3. Harbor pass-through, zero, mask, and preserved-timeout paths project the same reward and mask fields as before.
4. A legacy SkyRL-Gym environment produces byte-equivalent trainer fields through the adapter.
5. A native SkyRL-Gym verification result preserves verifier diagnostics and keeps raw outcome separate from a
   shaped optimization reward.
6. Token credit and disposition are projected in one place and cannot drift from loss masking.

Regression coverage spans the contract suite, SkyRL-Gym runner suite, Harbor error/result-processing suites, and
trajectory projection/processing suites.

## AIME migration boundary

AIME uses a native verifier implementing this protocol, with reward policy outside `AIMEEnv`. Its regression scope
includes `+1/-1` outcome buckets, failed-rollout length visibility, explicit unknown termination, and separation of
verifier correctness from length shaping. It does not add a second mitigation mechanism; obsolete AIME-specific
generation-to-reward hooks are removed as the shared contracts replace them.
