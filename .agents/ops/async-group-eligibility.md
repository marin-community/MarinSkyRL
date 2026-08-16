# Debugging log for async group eligibility

Prevent unusable rollout groups from reaching an async training step while preserving each advantage estimator's group-cardinality contract.

## Initial status

Fully async batch assembly rejects stale groups, retries their source prompts, releases their staleness-manager capacity, and waits for a complete replacement batch. Group content is not checked at this barrier. A fully loss-masked group can therefore reach `concatenate_trajectory_batches`, where behavior clipping rejects its missing rollout logprobs before the existing masked-position zero-fill can run.

Group semantics are implicit and split across configuration and estimator code. GRPO and RLOO consume every generated trial in a fixed-size group. RLOO-N permits baseline-ineligible trials but requires at least `trainer.algorithm.rloo_n_min_group_size` baseline-eligible trials. GAE, REINFORCE++, and the no-op distillation estimator do not compute group-relative advantages. Custom estimator registration currently carries no grouping metadata.

## Hypothesis 1

A resolved, mandatory group-advantage invariant can make async admission algorithm-aware without embedding estimator names in the barrier.

## Architectural design after review

### Estimator contract and run invariant

Every advantage estimator registration carries one immutable grouping contract variant:

```python
ExactPhysicalGroup()
MinimumBaselineEligibleGroup()
NoGroupAdvantage()
```

Registration requires this metadata for built-in and custom estimators. Concrete variants prevent unsupported cohort/cardinality combinations. The decorator and direct registration APIs reject missing contracts, so changing an estimator cannot silently inherit another estimator's semantics from Hydra defaults. The function registry continues to synchronize estimator functions through Ray; grouping metadata is required locally during configuration validation and is resolved before Ray workers start.

Every composed run also contains `trainer.algorithm.group_advantage_min_size`, which is null unless its estimator contract has `cardinality=MINIMUM`. Validation combines the registered contract, this user setting, and `generator.n_samples_per_prompt` into an immutable `GroupAdvantageInvariant` that is always present on the trainer:

- `EXACT/PHYSICAL`: the generated group must contain exactly `n_samples_per_prompt` physical trial rows. GRPO and RLOO use this contract.
- `MINIMUM/BASELINE_ELIGIBLE`: the physical group still contains exactly `n_samples_per_prompt` rows, but its advantage cohort may be ragged after baseline exclusions. At least `group_advantage_min_size` rows must have `exclude_from_baseline=false`. RLOO-N and RLOO-N-PBS use this contract.
- `NONE/NONE`: the estimator does not compute group-relative advantages. Group-cohort admission is bypassed. GAE, REINFORCE++, no-op, and uniform estimators use this contract.

Validation writes the resolved invariant into a canonical primitive `trainer.algorithm.resolved_group_advantage` config block before the config crosses the Ray driver boundary. The remote trainer constructs one frozen `GroupAdvantageInvariant` from that block and passes the same object to async admission, synchronous assertions, and RLOO-N advantage computation. The remote process never re-resolves estimator metadata, so the estimator and barrier cannot read different floors.

`rloo_n_min_group_size` is removed. `group_advantage_min_size` must be null for `EXACT` and `NONE`, and must satisfy `2 <= floor <= n_samples_per_prompt` for `MINIMUM`. Impossible contracts fail during configuration validation instead of retrying forever.

### Group inspection axes

Admission inspects independent facts rather than collapsing them into one “usable size”:

- `physical_count`: number of generated trial rows;
- `trainable_count`: rows whose loss mask contains at least one enabled token;
- `baseline_contributor_count`: rows with `exclude_from_baseline=false`.

Structural validation first requires aligned `response_ids` and `loss_masks`. A present `exclude_from_baseline` or rollout-logprob vector must also align. Missing `exclude_from_baseline` retains RLOO-N's current meaning: every physical row is baseline-eligible. Malformed producer output fails immediately and is never retried.

GRPO and RLOO require `physical_count == k`. Their existing advantage math includes all physical rows through `response_mask`, even when a row's policy loss is masked. This PR preserves that behavior: a partially masked `k`-row group remains legal, while the independent all-loss-masked rule rejects a group that cannot produce any gradient.

RLOO-N and RLOO-N-PBS require `physical_count == k` and `baseline_contributor_count >= floor`. Loss eligibility and baseline eligibility remain independent. An agent failure may be loss-masked but baseline-eligible and must continue contributing zero reward to the leave-one-out baseline. Infrastructure failures are baseline-ineligible and reduce the ragged advantage cohort. A fully loss-masked group is still rejected independently.

`NONE` bypasses group-cardinality and advantage-cohort checks, as required by estimators that do not use group-relative advantages, but still rejects fully loss-masked groups. The ordinary PPO backends retain their existing static batch geometry; custom flows that change physical cardinality remain responsible for producing a valid backend batch. General physical raggedness for group-relative estimators remains out of scope: UID construction, DP batch geometry, and worker gradient accumulation assume `prompt_count * n_samples_per_prompt`. Supporting it requires a separate dynamic-batching design and end-to-end backend tests.

Step-wise training emits transition rows rather than one row per trial. Structural alignment covers every emitted transition and validates final-row boundaries. Physical trial cardinality and baseline-cohort membership use final rows selected by `is_last_step`, matching the current group-advantage path. The all-loss-masked rule checks every emitted transition because an earlier transition can remain trainable when the final transition is masked.

### Extensible async admission

Introduce a pure async `GroupAdmissionPolicy` separate from staleness control and queue mutation. It structurally validates a completed `GeneratedOutputGroup`, then returns an `AdmissionDecision` with acceptance and an ordered tuple of retryable reasons. Initial retryable rules are:

1. staleness exceeds `max_staleness_steps`;
2. every trajectory is loss-masked;
3. the resolved group-advantage invariant is violated.

Producer-side staleness filtering remains in place so known-stale groups do not occupy the bounded completed queue. Barrier assembly evaluates the complete policy because buffered groups can become stale. Lifecycle-specific orchestration owns accounting: a producer-side stale rejection cancels its running submission slot; a barrier rejection owns an accepted slot and calls `on_rollouts_discarded`. Evaluation itself never mutates queues or counters.

Batch assembly scans every currently completed group. Each rejected group requeues `source_prompts`, releases accepted/submitted capacity, and cannot count toward `mini_batch_size`. Assembly waits until it has a full accepted batch and returns accepted surplus to the completed queue.

Metrics use `async/rejected_count`, `async/rejected_rate`, and `async/rejected_count/<reason>`. The rejection tuple records every applicable rule while its first entry supplies the primary routing reason. Staleness metrics continue to describe model-step age only.

The stall watchdog measures admitted progress, not merely completed generation. If completions have arrived but every one has been rejected for a full adaptive stall interval, batch assembly raises `GenerationStalledError` with reason counts even while generators remain alive. If no completion has arrived and generators remain alive, the existing deadline extension remains. This prevents a deterministic masked or below-floor prompt from resetting the progress clock forever without introducing a second retry-budget configuration.

### Logprob invariant

Admission does not accept fake behavior logprobs for trainable tokens when the policy objective hard-requires them. Structural validation checks rollout-logprob alignment whenever logprobs are present. Missing logprobs are an admission violation only when `policy_loss_requires_rollout_logprobs` is true, currently behavior clipping. Regular PPO may omit them. TIS retains its existing all-None graceful degradation instead of becoming a hard requirement in this PR. For behavior clipping, real values are mandatory on trainable positions and aligned zero placeholders are allowed only on masked positions. A fully masked group is retried before concatenation for every objective. The existing concatenation and behavior-clipping guards remain as defense-in-depth checks.

### Scope

This PR changes fully async admission and shared algorithm configuration/validation. Synchronous training receives the same resolved run invariant, but runtime replacement remains async-only because synchronous training has no equivalent retry barrier. A shared pre-advantage assertion makes a synchronous invariant violation fail with the structured reason instead of silently producing zero advantages. General synchronous regeneration remains future work.

### Regression coverage

- Configuration validation covers every built-in estimator contract, rejects missing registration metadata or contradictory floors, and preserves explicit custom-estimator contracts.
- Async batch assembly rejects and retries a fully masked fixed group, then waits for a usable replacement.
- Exact mode rejects a physical group with fewer than `k` rows and accepts a partially masked `k`-row group.
- Minimum mode accepts a baseline cohort at its floor and rejects one below it.
- RLOO-N counts baseline-eligible trials independently of loss masking.
- `none` mode bypasses cardinality and cohort checks but still rejects fully masked groups.
- A scan rejects every ineligible group in the completed buffer, including groups beyond the first candidate mini-batch, and capacity accounting releases every rejected attempt.
- Malformed output fails immediately rather than entering the retry queue.
- Rejected-only completions eventually raise the admitted-progress stall error instead of livelocking.
- Restored completed groups pass through the same admission scan and release capacity when rejected.

## Changes to make

Implement the reviewed design. Keep physical group cardinality fixed in this PR.

## Results

Three independent reviews covered algorithm semantics, async queue/accounting behavior, and configuration/API design. All three rejected the original single “usable size” abstraction. The revised design separates physical, trainable, and baseline-contributor counts; moves estimator semantics into registration metadata; retains fixed physical batches; separates fatal structural validation from retryable admission; and adds admitted-progress liveness.

All three reviewers passed the revised architecture. The focused admission, async barrier, stall detection, reward, and policy-optimization regressions pass. The full launcher and trainer CPU suite reached 1,495 passes and 19 skips; the three affected trainer fixtures were then corrected and pass. The remaining three failures are pre-existing HF-export fake-filesystem failures reproduced unchanged on `main`.

The mandatory Marin lint review identified and prompted corrections to the step-wise design text, rejection-result vocabulary, producer-side staleness comparison, stall-helper documentation, and a stale test name. Its remaining structural advisories are intentionally not applied:

- The resolved config block has a concrete default because unvalidated test and utility configs construct trainers directly; production validation always replaces it before Ray dispatch.
- Concrete fieldless contract variants are retained as the reviewed sum type. They prevent unsupported combinations from becoming representable; collapsing them into the runtime enum would undo the configuration review.
- Registry metadata is deliberately driver-local. Validation resolves it into the primitive `resolved_group_advantage` block before remote actors start; remote workers never look up registry metadata.
- Runtime structural checks remain at the Ray producer boundary even though callers are statically typed. TypedDict annotations do not validate deserialized or custom producer data.
- Admission inspection and barrier assembly keep their validation and condition-protected state transitions together so facts cannot drift between validation and accounting. Smaller helpers would expose partial states without reducing either responsibility.

## Future work

- [ ] Design dynamic backend batching before allowing physically ragged generated groups.
- [ ] Consider sharing async regeneration with a future synchronous retry barrier.
- [ ] Consider additional rejection rules, such as malformed verifier evidence, only when they have a defined regeneration policy.
