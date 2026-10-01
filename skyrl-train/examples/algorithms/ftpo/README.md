# Final-token preference optimization

FTPO trains full model weights on repetition boundaries from online rollouts.
The frozen reference starts from `trainer.ref.model.path`, which defaults to
`trainer.policy.model.path`: the model being repaired, not an external teacher.
Keep that original reference path when resuming a policy checkpoint.

Add these overrides to an existing Megatron/local-vLLM training configuration:

```bash
+algorithm_recipe=ftpo \
trainer.use_sample_packing=false \
trainer.policy.megatron_config.tensor_model_parallel_size=1 \
trainer.ref.megatron_config.tensor_model_parallel_size=1 \
trainer.policy.megatron_config.context_parallel_size=1 \
trainer.ref.megatron_config.context_parallel_size=1 \
generator.sampling_params.logprobs=32 \
+generator.engine_init_kwargs.max_logprobs=32
```

The recipe replaces reward advantages with an FTPO objective; reward grading may
still run for evaluation. It disables KL, entropy, reward-based dynamic sampling,
and off-policy correction. Use the existing data, rollout, optimizer, checkpoint,
and placement configuration. Provision the reference role even though KL is off.
Greedy generation (`temperature=0`) is supported. Candidate probabilities use
vLLM `raw_logprobs`; the objective uses unscaled logits regardless of sampling
temperature. Set `generator.engine_init_kwargs.max_logprobs` at least as high as
the requested K if the engine's configured limit is smaller.

The existing token-tail loop detector selects the start of the first repeated
copy in a periodic suffix. Its settings live under
`generator.trajectory_reward_shaping.loop`; reward shaping need not be enabled.
Only final, eligible trajectory rows contribute. Alternatives reuse exact token
IDs and probabilities captured by the existing rollout top-K path. FTPO filters
the repeated ID, case-equivalent decoded alternatives, short/blank alternatives,
and low relative probability candidates. It never re-tokenizes decoded text.
Full-response top-K capture is intentional; only one position per row trains.

`trainer.algorithm.ftpo` overrides the defaults in
[FTPOConfig](../../../skyrl_train/config/ftpo.py): margin 2, other-logit MSE 0.4,
target-logit MSE 0.05, target dead zone 0.5, relative min-p 0.01, and at most 20
chosen tokens. For example, `+trainer.algorithm.ftpo.margin=2.5` changes the margin.
The preference term is `softplus(margin - gap)` multiplied by
`clamp((margin - gap) / margin, 0, 1)`, where `gap` is chosen minus rejected logit.
The weight remains differentiable. Other vocabulary logits are tethered to the
frozen reference with MSE; chosen/rejected logits use a dead zone before MSE.
MSE normalization counts vocabulary entries across the complete optimizer step.

Frequency balancing is deterministic within each rollout batch: common rejected
tokens receive lower example weights, and common chosen tokens have occurrences
pruned with a seeded permutation. This differs from Antidoom's dataset-wide
balancing. The native token-tail detector also differs from its character-based
detector. These are deliberate online adaptations, not reproduction claims.

Training payloads reuse sequence tokens and `student_topk_indices`. New payloads
are `ftpo_chosen_mask` (`[B,T,K]`) and frozen `ftpo_reference_logits` (`[B,V]`).
The existing loss mask carries boundary weights. There is no `[B,T,V]` transport.
Empty batches and optimizer windows skip updates, including momentum and weight
decay. Optional `+trainer.algorithm.ftpo.early_stopping_chosen_win=0.95` stops when
the measured fraction of chosen alternatives beating their rejected token reaches
that threshold. This is a training metric, not a held-out quality gate. Stop state
is checkpointed; ordinary policy/optimizer and rollout resume mechanisms apply.

## Current limits and future work

**TP>1, context parallelism (CP), and sample packing are excluded.** Configuration
validation rejects them for this implementation. Supporting them requires
vocabulary-sharded candidate scoring and MSE reductions, plus correct boundary
indexing across CP and packed sequence layouts. These are important follow-ups,
not configuration switches that have merely been left untested.

The supported path is Megatron with TP=1, CP=1, no sequence parallelism, no sample
packing, and local vLLM top-K capture. Reference refresh callbacks, distillation,
and a critic are excluded. Full-vocabulary reference scores cost one vector per
trajectory; each model forward still produces ordinary vocabulary logits.
CPU tests cover the scalar loss/gradients, accumulation, token selection, padded
payload transport, and a tiny full-weight model update/resume. They do not replace
a real GPU Megatron/vLLM training run or demonstrate improved model quality.

Method source: [Antidoom](https://github.com/Liquid4All/antidoom), Apache-2.0,
revision `bd6a126476e18554b0cacaea3fd9f258fdde1f97`.
[Issue #883](https://github.com/marin-community/MarinSkyRL/issues/883) tracks the
broader implementation and evaluation work.
