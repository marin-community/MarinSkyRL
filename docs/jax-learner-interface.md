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
measure. The run's exact launch Git SHA is not present in the retained report.

The resolved `LearnerConfig` evidence for that exact run is:

| Field | E6 reproduction value | Evidence boundary |
| --- | --- | --- |
| Objective | `grpo` | Reported directly; this boundary only supports GRPO. |
| `policy_loss` | `regular` | Reported as unregularized GRPO; no DAPO or behavior-clipped objective is named. |
| `loss_normalization` | unknown | The report says advantages are standard-deviation normalized, which does not identify the policy-loss reduction. |
| `requires_reference_log_probs` | `false` | No KL term. |
| `clip_low`, `clip_high`, `dual_clip_ratio` | unknown | The resolved clip values are not retained in the report. |
| `reference_kl_coefficient`, `kl_estimator_type`, `use_absolute_kl` | coefficient absent; estimator and absolute-KL setting irrelevant and unresolved | The run has no KL term. |
| `use_rollout_importance_sampling` | `false` | TIS first appears as a later E10 ablation, not in E6. Its disabled cap is irrelevant and unresolved. |
| `update_epochs` | unknown | The report does not retain the resolved value. |
| `logprob_temperature` | `1.0` | Reported directly for the campaign's 256 prompts × 16 responses setup. |
| `max_sequence_length` | `8192` | Reported as the E6 request window. |

The same evidence fixes FSDP2, expert parallelism 1, no entropy term, and no sample packing. Unknown values stay
explicit here so the next parity goal can recover the launch artifact rather than silently substitute current defaults.

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
such as an all-zero loss mask, without advancing the policy version. Exceptions are failures and never advance fake
state. A backend that has partially mutated durable state before an exception must enter `failed` lifecycle state and
remain unusable until a checkpoint restore. A successful update makes the installed inference version outdated.

`LearnerState` is the authority for publication and checkpoint state. `pending` and `failed` publications are not
rollout-ready. MSRL resumes generation only when the installed and current policy versions match and
`LearnerState.ready_for_rollouts` is true.

A learner checkpoint contains resumable model, optimizer, RNG, and learner-step state. It is different from a weight
snapshot installed in inference. MSRL stages that state with its trainer, dataloader, data-consumption, and async-buffer
artifacts. It atomically writes the latest checkpoint marker and begins retention cleanup only after every required
save callback succeeds. Loading a learner checkpoint deliberately clears the fake receiver's installed version; an
initial publication must reconcile inference before generation. Exact replay of arbitrary in-flight async work is
outside this learner contract and remains an MSRL concern.

The deterministic CPU fake lives under
[`skyrl_train.testing`](../skyrl-train/skyrl_train/testing/stateful_fake_learner.py). Its log probabilities depend on
tokens and fake parameters, updates depend on masked advantages and fake state, and checkpoints restore the next
update exactly. Tests can request one-shot log-probability, update, publication, save, load, or close failures, and can
leave a publication pending. This behavior is only an orchestration oracle; it does not model real GRPO gradients.

## Levanter mapping

The mapping below was checked read-only at `marin` revision
[`e49f36f2d7434776d9289a6bf3781f22b419ba39`](https://github.com/marin-community/marin/tree/e49f36f2d7434776d9289a6bf3781f22b419ba39).

| Responsibility | Reusable Levanter operation | Remaining adapter glue | Runtime assumption still unresolved |
| --- | --- | --- | --- |
| Initialize model, optimizer, RNG, step, mesh, and sharding | [`Trainer`](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/trainer.py#L269) and [`TrainerState`](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/trainer_state.py#L38) already own these objects. | Build one long-lived learner service, lower `LearnerConfig`, and reject unsupported Snowball options before JAX allocation. | Iris/JAX process startup and all-rank service ordering have not been exercised through this interface. |
| Current and reference token log probabilities | `LmHeadModel.compute_next_token_loss(reduction=None)` exposes per-position cross entropy, and Snowball has an HF converter. | Convert unpacked NumPy rows to named arrays, apply temperature, select the shifted response tokens, and retain a separate reference state when requested. | Snowball `activations` still [ignores its supplied attention mask](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/models/snowball.py#L703). One example per row prevents cross-example packing leakage, but padded-token behavior needs a CPU JAX check. |
| One GRPO-oriented update | [`Trainer.train_step`](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/trainer.py#L517) accepts a JIT-able loss; [`TrainerState.take_step`](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/trainer_state.py#L126) updates optimizer, model, RNG, and optimizer-step state. | Implement the selected PPO/behavior clipping, optional TIS and KL term, masks, and exact MSRL normalization as a Levanter loss. MSRL `global_step` names orchestration progress. The adapter owns one `update_count` and `policy_version` per complete `UpdateRequest`, even if update epochs require several Levanter optimizer steps. Persist both adapter counters beside `TrainerState`, advance them only after the whole request commits, and enter failed lifecycle state after partial mutation until restore. | No value or gradient parity has been established between this loss and MSRL for Snowball. The transaction boundary around several Levanter steps has not been exercised. |
| Publish a version to vLLM | Snowball's `HFCheckpointConverter` maps Levanter parameters to HF names, and [`save_pretrained`](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/compat/hf_checkpoints.py#L1055) already plans bounded shards. | Gather or stream a complete stable HF-named snapshot, preserve dtype and router bias, call MSRL's live vLLM receiver, and update `LearnerState` only after installation. Tag every served async segment with that installed version; reject a trajectory that spans versions. | HF disk export does not prove a memory-safe, atomic, low-latency live publication path for sharded Snowball, and the serving tag has no live vLLM implementation yet. |
| Save and restore training state | Levanter's [`save_checkpoint`](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/checkpoint.py#L781) is JAX-array aware; [`load_checkpoint`](https://github.com/marin-community/marin/blob/e49f36f2d7434776d9289a6bf3781f22b419ba39/lib/levanter/src/levanter/checkpoint.py#L897) restores against an exemplar tree. | Make MSRL wait for Levanter's async checkpoint commit before writing its latest marker, restore `LearnerState`, and force inference reconciliation. | Multi-host completion, failure propagation, and restart behavior have not been tested with an MSRL checkpoint directory. |
| Cleanup | `Trainer.__exit__` waits for in-flight checkpoint serialization and closes tracker and mesh contexts. | Tie service shutdown to MSRL teardown and surface cleanup failure without reporting a successful unfinished operation. | Ray actor and JAX process termination ordering has not been exercised. |

The smallest next implementation test is a single-process JAX CPU test with a tiny Snowball-shaped model and two
unpacked, differently padded rows. It should compute response log probabilities, take one masked GRPO update, save and
restore `TrainerState`, and reproduce the next update exactly. That test should compare log probabilities, loss, and
gradients against an independent small Torch or NumPy reference. Live vLLM publication and any accelerator execution
come later.
