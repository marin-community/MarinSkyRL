# JAX-first learner interface

MSRL can drive a learner without exposing the learner's model type, optimizer, sharding, or collective layout. The
contract is defined in [`learner.py`](../skyrl-train/skyrl_train/learner.py). The existing Torch worker groups remain the
default implementation path. A caller must supply a learner explicitly; MSRL never selects the stateful fake when an
operation is missing.

Pass a `Learner` implementation through `RayPPOTrainer(..., learner=learner)` or
`FullyAsyncRayPPOTrainer(..., learner=learner)`. MSRL validates the configuration before calling `initialize` and does
not allocate Torch training actors. An entrypoint can instead pass an uninitialized learner to `BasePPOExp`; its
side-effect-free preflight runs before tokenizer, dataset, placement-group, inference, trajectory, or learner allocation.
This path currently accepts GRPO with the regular or behavior-clipped policy loss, the four existing loss reductions,
optional reference log probabilities, and optional rollout importance sampling. It rejects critic training, sample
packing, step-wise training, entropy loss, learner/inference colocation, mutable reference policies, think-token loss
weighting, MoE router replay, z-clip, and stale-clip.

## Workload boundary

The concrete baseline is Ben Feuer's representative E6 reproduction run A,
`rl-snowball-e6-rno2a-rlvrmath-grug-67b-a-20260812-150653-98cb43`, through step 20. It is pinned by the completed
[Snowball math campaign](https://github.com/marin-community/marin/issues/7786) and the versioned [August 27
report](https://storage.googleapis.com/marin-public/benjaminfeuer/snowball-67b-a2b-math-rl/2026.08.27.1/index.html).
The report identifies this reproduction as the representative comparator; reward alone is not a held-out quality
measure. The retained run log records MarinSkyRL source
[`1f7ed486beef9566a6077de49425f483148f1ff8`](https://github.com/marin-community/MarinSkyRL/tree/1f7ed486beef9566a6077de49425f483148f1ff8)
and the full resolved configuration.

The resolved `LearnerConfig` evidence for that exact run is:

| Field | E6 reproduction value | Evidence boundary |
| --- | --- | --- |
| Objective | `grpo` | Reported directly; this boundary only supports GRPO. |
| `policy_loss` | `regular` | Reported as unregularized GRPO; no DAPO or behavior-clipped objective is named. |
| `loss_normalization` | `token_mean` | Resolved run configuration. |
| `requires_reference_log_probs` | `false` | No KL term. |
| `clip_low`, `clip_high`, `dual_clip_ratio` | `0.2`, `0.2`, irrelevant for regular GRPO | Resolved run configuration. |
| `reference_kl_coefficient`, `kl_estimator_type`, `use_absolute_kl` | coefficient absent; estimator and absolute-KL setting irrelevant and unresolved | The run has no KL term. |
| `use_rollout_importance_sampling` | `false` | TIS first appears as a later E10 ablation, not in E6. Its disabled cap is irrelevant and unresolved. |
| `update_epochs` | `1` | Resolved run configuration. |
| `logprob_temperature` | `1.0` | Reported directly for the campaign's 256 prompts × 16 responses setup. |
| `max_sequence_length` | `8192` | Reported as the E6 request window. |

The same evidence fixes FSDP2, expert parallelism 1, no entropy term, no sample packing, a frozen router bias, and
AdamW at learning rate `1e-5`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, maximum gradient norm `0.5`,
and a zero-warmup constant schedule. The [Levanter Snowball reference](levanter-snowball-training.md) records the
recovered artifacts, the implemented subset, and the real-GPU evidence.

The interface also carries three scenarios that change the boundary:

- an optional reference-policy forward for KL loss or reward shaping;
- rollout log probabilities and one behavior-policy version per row for delayed asynchronous batches; and
- resumable learner state followed by explicit inference-policy reconciliation.

Luke Lee's [R2E-Gym report](https://github.com/marin-community/marin/issues/9114) establishes an asynchronous Snowball
scenario with staleness two. It does not identify the resolved KL setting, so it does not define a combined async+KL
recipe. The interface supports the two concerns independently.

## Batch and probability meanings

Every `LearnerBatch` row is one independent example. MSRL does not pack examples for this path. `sequences` and
`attention_mask` have shape `[batch, sequence]`. The response is the trailing `response_length` positions.
`response_mask`, `loss_mask`, and every log-probability array have shape `[batch, response_length]`.

`response_mask` marks real response tokens and excludes right padding. `loss_mask` may further exclude invalid or
untrainable response tokens and must be zero wherever `response_mask` is zero. Padding rows added for data parallelism
are unnecessary because the learner receives the whole batch. The configured normalization is part of `LearnerConfig`;
the learner must apply the same token, sequence, fixed-length, or global denominator rule as MSRL.

The four probability channels are distinct:

- **Rollout** log probabilities came from the inference policy while tokens were sampled. In an async batch, rows may
  name older behavior-policy versions. The serving adapter must attach the installed learner version to every inference
  segment. MSRL accepts a row only when all of its segments name one version, then preserves one version per response
  row. It never derives this identity from `global_step`; the separate rollout step remains the key for staleness
  admission. The live vLLM tagging adapter is still follow-up work, so a real async learner run fails if these serving
  observations are absent.
- **Old-policy** log probabilities are recomputed by the learner immediately before the update. They name one stable
  learner policy version and are the PPO clipping baseline for the regular objective.
- **Current-policy** log probabilities are recomputed with gradients inside the update. They are not supplied by MSRL.
  A real learner must compare them with the old or rollout channel according to the selected objective.
- **Reference** log probabilities come from a fixed reference policy. They are absent when no KL-dependent feature is
  configured; reference-policy updates are rejected by this boundary.

The Torch bridge copies CPU tensors into NumPy arrays once. This keeps Torch transport in MSRL without requiring Torch
autograd or Torch model objects inside a JAX learner.

## State transitions and ownership

`initialize` rejects unsupported features before learner resources are created. MSRL owns rollouts, queues, staleness
admission, rewards, advantages, callback progress, and dataloader state. The learner owns model, optimizer, RNG, update
step, JAX meshes, sharding, collectives, and framework-specific checkpoint data.

An update returns `succeeded` only after state changes are complete. It returns `skipped` for a valid no-update batch,
such as an all-zero loss mask, without advancing the policy version. Exceptions are failures and never advance learner
state. A backend that has partially mutated durable state before an exception must enter `failed` lifecycle state and
remain unusable until a checkpoint restore. A successful update makes the installed inference version outdated.

`LearnerState` is the authority for publication and checkpoint state. `pending` and `failed` publications are not
rollout-ready. MSRL resumes generation only when the installed and current policy versions match and
`LearnerState.ready_for_rollouts` is true.

A learner checkpoint contains resumable model, optimizer, RNG, and learner-step state. It is different from a weight
snapshot installed in inference. MSRL stages that state with its trainer, dataloader, data-consumption, and async-buffer
artifacts after removing any stale completion marker. After every required save callback succeeds, it writes the
step-directory completion marker, atomically advances the root latest marker, and then begins retention cleanup. The
Levanter path requires that completion marker and dataloader state on restore. Loading a learner checkpoint deliberately
clears the installed inference version; an initial publication must reconcile inference before generation. Exact replay
of arbitrary in-flight async work is outside this learner contract and remains an MSRL concern.

The deterministic CPU fake lives under
[`skyrl_train.testing`](../skyrl-train/skyrl_train/testing/stateful_fake_learner.py). Its log probabilities depend on
tokens and fake parameters, updates depend on masked advantages and fake state, and checkpoints restore the next
update exactly. Tests can request one-shot log-probability, update, publication, save, load, or close failures, and can
leave a publication pending. This behavior is only an orchestration oracle; it does not model real GRPO gradients.

## Concrete Levanter mapping

The optional implementation lives in `skyrl_train.learners.levanter_snowball`. It narrows the general interface to the
single synchronous E6-derived workload documented in [Levanter Snowball training](levanter-snowball-training.md).

| Responsibility | Implementation | Evidence |
| --- | --- | --- |
| Initialize model, optimizer, RNG, mesh, and sharding | Levanter `Trainer`, `TrainerState`, the Snowball HF converter, and a one-node data mesh | A two-H100 learner task loaded a real tiny Snowball HF checkpoint and completed repeated sharded optimizer steps. |
| Token log probabilities and GRPO | Unpacked rows are compacted independently, response predictor positions are retained, and Levanter averages each sequence's masked token mean across devices and accumulation steps, matching E6's one-sequence GPU microbatches | Independent PyTorch comparisons cover unequal response lengths, log probabilities, loss, gradients, and the first AdamW update. Padding and two-device microbatch tests cover alignment and normalization. |
| Publish a version to vLLM | The learner converts all parameters to HF names, sends Gloo chunks, and requires receipts with the exact source-name and installed-parameter counts and digests before advancing the installed version | The three-H100 gate reads dense, expert, language-head, and FP32 router-bias values back from the active vLLM worker after each final update and then generates. A chunk respects its byte limit unless one tensor alone exceeds it. |
| Save and restore training state | Levanter TensorStore checkpoints contain model, optimizer, RNG, step, and policy version; MSRL writes a step-directory completion marker before updating the root latest marker, after learner, trainer, and dataloader state commit | A fresh Ray runtime and learner restore all 306 arrays byte exactly, reconcile vLLM before generation, replay the saved next batch within the accepted `1e-4` GPU bound, and resume the public trainer. The next-update maxima were `1.4142e-5` for parameters and `2.5635e-5` for optimizer state. Missing completion markers or dataloader state reject restore. |
| Export and cleanup | The HF converter can export the current in-memory Levanter model after a completed checkpoint; trainer shutdown closes the learner and inference actors | CPU interface tests cover the converter call, incomplete-checkpoint rejection, and failure visibility. The GPU gate tears down both fresh processes, but it does not qualify an exported directory in a separate production consumer. |

Async rollout tagging, KL or reference policies, sample packing, learner/inference colocation, other model families,
multi-host execution, and full-size Snowball qualification remain outside this concrete implementation.
