# Levanter Snowball training

The `skyrl_train.entrypoints.levanter_snowball` entrypoint runs one synchronous Snowball GRPO workload with MSRL
orchestration and a Levanter learner. MSRL owns generation, rewards, advantages, and progress. Levanter owns the model,
optimizer, random key, step, mesh, sharding, collectives, checkpoint, and Hugging Face export.

This integration has been exercised with a small Snowball model on two H100 learner GPUs and one separate H100 vLLM
GPU. It does not qualify the 67B-A2B checkpoint, more than one learner node, or a training campaign.

## Supported workload

Configuration validation runs before JAX or Torch is imported and before GPUs or inference actors are allocated. The
entrypoint accepts this boundary:

- one learner node, with the data mesh spread across its GPUs;
- separate local asynchronous vLLM inference engines;
- synchronous GRPO with `policy_loss_type=regular`, `loss_reduction=token_mean`, clipping `0.2/0.2`, group standard
  deviation normalization, no batch advantage normalization, and one update epoch;
- no KL, entropy, TIS, critic, sample packing, or step-wise training;
- restored dataloader state, no LoRA, one train and forward example per GPU microbatch, a generated trajectory batch
  divisible by the learner GPU count, frozen Snowball router bias, and expert parallelism 1;
- AdamW with learning rate `1e-5`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, maximum gradient norm `0.5`,
  zero warmup, and a constant schedule;
- Levanter reference attention, ring MoE dispatch, Gloo weight publication, vLLM pipeline parallelism 1, and sampling
  temperature `1.0`.

Other settings fail preflight. They do not fall back to a Torch learner or the stateful test fake.

Install the locked runtime with:

```bash
uv sync --frozen --extra cuda --extra vllm --extra levanter-gpu --group dev
```

Launch the entrypoint through the same Hydra configuration path used by `main_base`, replacing the entrypoint with:

```text
skyrl_train.entrypoints.levanter_snowball
```

Set `trainer.policy.levanter.*` for Levanter dtypes, reference kernels, publication chunks, timeout, and log directory.
The defaults are in `skyrl-train/skyrl_train/config/ppo_base_config.yaml`. The pinned Levanter revision is
[`e49f36f2d7434776d9289a6bf3781f22b419ba39`](https://github.com/marin-community/marin/tree/e49f36f2d7434776d9289a6bf3781f22b419ba39).
The GPU extra installs JAX 0.11.1 with CUDA 12, matching the Torch and vLLM runtime. It omits Levanter's optional
Quack/CUTLASS GPU set because its versions conflict with the vLLM closure and this implementation selects Levanter's
reference kernels.

## E6 reference and tested derivative

The learning reference is E6 reproduction run A,
`rl-snowball-e6-rno2a-rlvrmath-grug-67b-a-20260812-150653-98cb43`. Its retained log records MarinSkyRL source
[`1f7ed486beef9566a6077de49425f483148f1ff8`](https://github.com/marin-community/MarinSkyRL/tree/1f7ed486beef9566a6077de49425f483148f1ff8)
and contains the [resolved configuration](https://huggingface.co/datasets/penfever/snowball-67b-a2b-math-rl-artifacts/blob/ceb7826b33085c3f3f1ff86538c2e7a39a2dc216/artifacts/e6/repro/finelog_run-A_20260812-150653-98cb43.log).

| Setting | E6 reproduction A | H100 integration gate |
| --- | --- | --- |
| Model | Snowball 67B-A2B, `marin-community/grug-67b-a2b-sft-s2-thinking-step630` | Local 5-layer Snowball, hidden size 64, 8 experts, top 2, vocabulary 64 |
| Model revision | `6808fe5c219471517bd51df35addefd38ebebf89` | Locally generated from seed 17 |
| Data | `s3://marin-us-east-02a/iris/rl-data/snowball-67b-a2b-rlvrmath-7498/train.parquet` | Four distinct 6-token prompts; each iteration consumes two prompts and samples two responses per prompt |
| Request geometry | 1,664 prompt + up to 6,528 generated tokens; window 8,192 | 6 prompt + 4 generated tokens; window 128 |
| GRPO geometry | Batch 256, 16 responses per prompt | Batch 2, 2 responses per prompt |
| Objective | Regular clipped GRPO, token mean, clip `0.2/0.2`, one update epoch, no KL/entropy/TIS | Same |
| Optimizer | AdamW, LR `1e-5`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, max norm `0.5`, zero-warmup constant schedule | Same |
| Router bias | Frozen | Same |
| Dtypes | BF16 model dtype; FP32 router bias | FP32 parameter storage, BF16 compute and serving weights, FP32 outputs and router bias |
| Learner topology | FSDP2, EP1, 8 nodes × 8 H100 | Levanter data mesh, EP1, 1 node with 2 active H100s |
| Inference topology | 2 engines, each TP1/DP8/EP8 | 1 engine, TP1/DP1/EP1 on one separate H100 |
| Weight transport | NCCL | Gloo, with BF16 model tensors and FP32 router biases |
| Updates | 20 | Two public-trainer updates, then a fresh process restores the step-1 checkpoint and repeats the saved second update |

The retained run log resolves a regional cache directory named
`a822321c2c21af099189e7116104b3cf5142c119`. That identifier is not a revision in the public model repository. A later
full-checkpoint diagnostic read and hashed all 39 cached weight shards, totaling 134,157,827,064 bytes. Every SHA-256
matched the Git LFS object in public revision `6808...`. The cache directory name was stale; the weight identity is now
resolved to the public revision above.

## Weight publication and resume

Every successful learner update invalidates the installed inference version. Publication converts the complete
Levanter state to Hugging Face names and sends chunks through MSRL's existing named-weight receiver. Each chunk stays
within the configured byte threshold except when one tensor alone exceeds it. Snowball
expert matrices stay stacked because the Grug vLLM loader unbinds their expert dimension into its fused storage. Router
bias tensors stay FP32; other tensors use the configured generator dtype.

The learner marks a version installed only after every active vLLM worker returns a receiver-observed receipt. The
receipt must match the complete source weight name count and digest, the expected top-level vLLM parameter count and
digest, and completion of vLLM's layer-wise reload. Standard Qwen q/k/v and gate/up sources are loaded separately so a
packed target name cannot hide a skipped sibling. The GPU gate also reads dense, expert, language-head, and router-bias
values back from vLLM after each final update. Complete per-expert-slice acknowledgement still belongs in the full-size
multi-host gate.

MSRL writes its completion marker only after Levanter's TensorStore checkpoint has committed. The payload includes the
full model, Adam state, Levanter training key and step, and the learner policy version. The trainer stages learner,
trainer, and dataloader state after deleting any stale marker. It atomically writes the step-directory completion
marker before advancing `latest_ckpt_global_step.txt`; restore requires the marker and dataloader state. Loading
clears the installed version. Generation remains blocked until the restored learner republishes and the vLLM receipts
complete. An incomplete callback save cannot be loaded or exported.

The fresh-process gate restarts Ray, creates a new learner and vLLM process, and compares every checkpoint array before
publication. All 306 arrays restore byte exactly. The CPU subprocess restores exactly, including its next update in
the current CPU runtime. On H100, the accepted absolute replay tolerance is `1e-4` for model and optimizer arrays. The
measured next-update maxima were `1.4141202e-5` for model parameters and `2.5634421e-5` for optimizer state; step and
random-key metadata remained exact.

The implementation can export the current in-memory model with Levanter's Hugging Face converter after a completed
checkpoint. Unit tests cover the call and reject export from an incomplete checkpoint. The real-GPU gate did not load
that exported directory into a separate production-sized consumer, so export interoperability beyond the converter is
not qualified here.

## Validation evidence

The independent CPU oracle builds the same tiny model in the native PyTorch Grug implementation. It compares selected
response log probabilities, the masked loss, every gradient, and the first AdamW update:

| Quantity | Mean absolute difference | Maximum absolute difference |
| --- | ---: | ---: |
| Token log probability | `9.5367433e-8` | `2.3841858e-7` |
| Masked GRPO loss | `0` | `0` |
| Gradient | `5.0648888e-11` | `8.9406967e-8` |
| First AdamW update | `4.0966097e-10` | `1.1920929e-7` |

`skyrl-train/tests/cpu/test_levanter_snowball_learner.py` also covers different left-padding widths, right-padded
responses, response predictor positions, per-sequence masked token means averaged across devices and accumulation,
frozen router bias, stacked expert publication names, bounded probes from a two-device sharded array, and fresh-process
checkpoint replay. This module requires the optional `levanter` extra and is skipped by the default PR CPU job; run it
with the focused command below before changing the numerical path.

```bash
uv run --frozen --extra cuda --extra vllm --extra levanter-gpu --group dev \
  pytest skyrl-train/tests/cpu/test_levanter_snowball_learner.py
```

The opt-in real-GPU gate is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run --frozen \
  --extra cuda --extra vllm --extra levanter-gpu --group dev \
  pytest -s skyrl-train/tests/gpu/levanter_snowball_cycle.py
```

It runs the real `TrajectoryRunner` and public `RayPPOTrainer.train` path. Phase one performs initial publication, two
stochastic rollout/update/publication cycles, exact vLLM readbacks after each final update, and callback checkpoints at
steps 1 and 2. Phase two starts a fresh Ray process, restores step 1 through the public trainer, verifies exact
checkpoint arrays and dataloader position, republishes before generation, replays the saved second batch, and completes
the public second update. The resumed trajectory UIDs and response token IDs exactly match the saved batch.

The tiny-model gate was Iris job `/romain/dev-gpu-levsnow-fix-01a09cd0`. It passed in 155.32 seconds of test time. The
whole-node allocation lasted 22 minutes 45.79 seconds, or 3.035 allocated H100-hours, with three of the eight H100s
active. Accounting across eight allocation incarnations, including an earlier run hidden by a reused job name, gives a
conservative corrected total of 7.188 H100-hours.

| Measurement | Seconds |
| --- | ---: |
| Learner initialization | `3.353`–`3.370` |
| First forward compilation | `1.603` in phase one; `2.010` after restart |
| First complete update | `9.892` in phase one; `10.046` for fresh-process replay |
| Warm complete update | `0.0423` |
| Warm forward | `0.0113` |
| Warm full iteration | `0.577` |
| Compiled full iteration | `12.708` in phase one; `12.634` after restart |
| Checkpoint commit and MSRL marker | `0.192`–`0.243` |
| Fresh-process checkpoint load | `0.369` |
| Weight publication | `0.156`–`0.242` |

Each iteration consumed 24 prompt tokens and generated 16 response tokens from two prompts and four trajectories. The
learner batch was four rows, one example per GPU microbatch. The data mesh had size two; the measured batch had two
addressable `[2, 8]` shards. A `[128, 64]` query weight used `P('model', 'data')` and had two addressable `[128, 32]`
shards. Timings include device synchronization before the learner reports completion. They describe this tiny
reference-kernel gate and do not predict 67B throughput.

## Full-size inference diagnostic

A later inference-only diagnostic loaded the pinned 67B-A2B checkpoint on one eight-H100 node with vLLM TP1/DP8/EP8.
It used the exact public config and tokenizer plus the 39 byte-verified cached weight shards. Model loading took 3.50
seconds and 18.35 GiB per rank. The initialized server occupied 71,175-71,273 MiB on each H100.

On a deterministic 128-example sample of `HuggingFaceH4/MATH-500` revision
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, the initial policy produced 346,374 tokens in 75.02 seconds, or 4,617
output tokens/second across the node. The frozen evaluator was the recovered E6 `AIMEVerifier`. It accepted 10/128
responses and parsed 17/128, for mean reward -0.84375. Many equivalent boxed answers did not use its exact trailing
`Answer:` form.

A post-hoc `math-verify==0.8.0` answer-equivalence audit scored 100/128 answers correct (78.1%, Wilson 95% interval
70.2%-84.4%). Ninety correct answers also completed normally; ten correct answers appeared before a later length stop.
Overall, 96/128 responses completed normally and 32/128 reached the 6,528-token limit. This exploratory audit was not
the frozen evaluator, and neither result is JAX learning evidence. The fixed sample is a pilot cohort rather than a
complete MATH-500 result, and the timing is an inference measurement rather than an MSRL iteration.

## Remaining scope

The full checkpoint and DP8/EP8 inference now work, but no full-size Levanter learner was initialized. Multi-host
checkpoint completion, learner collectives, a real update and publication, campaign learning quality, learner
throughput, and end-to-end throughput remain untested. The small model uses reference attention and ring MoE, so its
timings are diagnostic. There is no matched Megatron measurement for this geometry.

The tiny end-to-end GPU gate predates the final CUDA 12 dependency selection and allocator-policy cleanup. The final
dependency closure has an eight-H100 JAX collective smoke, not a Levanter update. The next full-size gate must also
subsume that end-to-end runtime validation.

With the currently resolved `marin-iris` package, Levanter cannot initialize its direct Iris metrics writer because
that package lacks `iris.runtime.telemetry.resolve`. Levanter catches this failure and training continues; MSRL logs and
the returned learner metrics remain available. Direct Levanter telemetry needs a compatible Iris package in a follow-up.

The next expensive check is one finite 67B update on a credible multi-host learner topology, followed by checkpoint,
restart, and sharded publication to a separate vLLM node. It should add source-component acknowledgement or exhaustive
readback for fused expert slices. The small gate has already established orchestration, numerical semantics, local
sharding on two GPUs, live installation, and resume mechanics. A full-model run is not needed to review this scoped
integration.
