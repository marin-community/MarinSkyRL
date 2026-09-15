# Levanter Snowball training

The `skyrl_train.entrypoints.levanter_snowball` entrypoint runs one synchronous Snowball GRPO workload with MSRL
orchestration and a Levanter learner. MSRL owns generation, rewards, advantages, and progress. Levanter owns the model,
optimizer, random key, step, mesh, sharding, collectives, checkpoint, and Hugging Face export.

An earlier feasibility run completed two Snowball 67B-A2B steps on four eight-H100 JAX learner hosts with a separate
eight-H100 vLLM TP1/DP8/EP8 host. It predates the corrected objective and stronger publication checks, so it does not
qualify the current revision. The current small gate covers two real update/generation cycles and DP2/EP2 publication;
target-size qualification is recorded separately when it completes.

## Supported workload

Configuration validation runs before JAX or Torch is imported and before GPUs or inference actors are allocated. The
entrypoint accepts this boundary:

- one or more learner nodes, with one JAX process per node and the data mesh spread across all learner GPUs;
- separate local asynchronous vLLM inference engines;
- synchronous GRPO with `policy_loss_type=regular`, `loss_reduction=token_mean`, clipping `0.2/0.2`, group standard
  deviation normalization, no batch advantage normalization, and one update epoch;
- no KL, entropy, TIS, critic, sample packing, or step-wise training;
- restored dataloader state, no LoRA, one train and forward example per GPU microbatch, a generated trajectory batch
  divisible by the learner GPU count, frozen Snowball router bias, and expert parallelism 1;
- AdamW with learning rate `1e-5`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, maximum gradient norm `0.5`,
  zero warmup, and a constant schedule;
- Levanter reference or GPU FA4 attention, ring MoE dispatch, Gloo weight publication, vLLM pipeline parallelism 1, and sampling
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

Set `trainer.policy.levanter.*` for Levanter dtypes, kernels, publication chunks, timeout, and log directory.
The defaults are in `skyrl-train/skyrl_train/config/ppo_base_config.yaml`. The pinned Levanter revision is
[`779cd403521e02d1615d8c49290cd8de63efdde0`](https://github.com/marin-community/marin/tree/779cd403521e02d1615d8c49290cd8de63efdde0).
The GPU extra installs JAX 0.11.1 with CUDA 12, matching the Torch and vLLM runtime. The 67B run used segmented GPU
FA4 attention and ring MoE dispatch; reference attention remains useful for the tiny numerical gate.

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
| Inference topology | 2 engines, each TP1/DP8/EP8 | 1 engine, TP1/DP2/EP2 on two separate H100s |
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

vLLM's layer-wise reload temporarily places parameters that have not arrived yet on the meta device. The learner first
pauses every EngineCore scheduler, then opens that reload bracket. This also stops data-parallel dummy batches, which
otherwise can execute against the incomplete model and kill the engine. Generation resumes only after every receipt
passes and the prefix cache resets. Any ambiguous failure leaves generation paused and the learner failed. At an idle
synchronous boundary, the client skips its five-second request-drain grace but still waits for the scheduler-level pause;
active requests retain the grace period.

The learner marks a version installed only after every active vLLM worker returns a receiver-observed receipt. The
receipt must match the complete source weight name count and digest, the expected top-level vLLM parameter count and
digest, and completion of vLLM's layer-wise reload. Standard Qwen q/k/v and gate/up sources are loaded separately so a
packed target name cannot hide a skipped sibling. Grug expert tensors are loaded one projection and expert at a time;
the receiver records only slices owned by that EP worker. Publication requires the union across workers to contain
every expected slice exactly once. A negative CPU test drops one local slice and requires publication to fail. The
original 67B run predates this stronger receipt, so its eight-worker success proves top-level installation but does not
by itself qualify the new per-slice receipt.

MSRL writes its completion marker only after Levanter's TensorStore checkpoint has committed. The payload includes the
full model, Adam state, Levanter training key and step, and the learner policy version. The trainer stages learner,
trainer, and dataloader state after deleting any stale marker. It atomically writes the step-directory completion
marker before advancing `latest_ckpt_global_step.txt`; restore requires the marker and dataloader state. Loading
clears the installed version. Generation remains blocked until the restored learner republishes and the vLLM receipts
complete. An incomplete callback save cannot be loaded or exported.

The fresh-process GPU gate restarts Ray, creates a new learner and vLLM process, and compares every checkpoint array before
publication. All 306 arrays restore byte exactly. The CPU subprocess restores exactly, including its next update in
the current CPU runtime. On H100, the accepted absolute replay tolerance is `1e-4` for model and optimizer arrays. The
measured next-update maxima were `1.4141202e-5` for model parameters and `2.5634421e-5` for optimizer state; step and
random-key metadata remained exact.

A separate two-process CPU test now stops both JAX processes after saving, starts two new processes, restores every
model, optimizer, RNG, and step leaf byte exactly on both ranks, and completes the next collective update. The target
67B job logged successful atomic TensorStore commits and MSRL completion markers, but the retained environment could
not list or restore those S3 objects from the devbox.

The implementation can export the current in-memory model with Levanter's Hugging Face converter after a completed
checkpoint. Unit tests cover the call and reject export from an incomplete checkpoint. The real-GPU gate did not load
that exported directory into a separate production-sized consumer, so export interoperability beyond the converter is
not qualified here.

The two-step target configuration sets the checkpoint interval beyond the run length. `CheckpointCallback` still saves
once at train end, so no per-step TensorStore write occurs and one final native checkpoint remains. HF export is
disabled explicitly. Publication and checkpoint timings are reported separately.

## Validation evidence

The earlier Iris feasibility run `/romain/levanter-snowball-real-01a09cd0-r17` used the pinned 67B model, 32 learner H100s, and a separate
eight-H100 vLLM host. All five tasks exited successfully with no retries or preemptions. Step 1 generated 417,012
response tokens, completed a finite update with a nonzero parameter probe, published policy version 1, committed its
native checkpoint, exported BF16 weights, and then generated 714,857 tokens with the installed policy. Step 2 also
updated, published, checkpointed, and exported. That run predates the corrections below and is feasibility evidence,
not current numerical qualification.

The earlier run exposed BF16 replay drift between two forward passes of the unchanged policy: maximum absolute
log-probability differences were 6.85 and 7.47, affecting 0.198% and 0.529% of selected tokens at the PPO clip bounds.
The rejected same-forward workaround hid this discrepancy by replacing supplied old-policy scores with
`stop_gradient(current_scores)`. The corrected objective preserves independent old scores and computes real ratios.

The main cause was local MoE output combine: scatter-add collisions used GPU atomics, so repeated unchanged-policy
scores could differ. Levanter now gathers each token's top-k expert outputs by reverse dispatch position and sums them
in a fixed order. Repeated scoring on the representative 8-H100 shape became bit exact. A smaller deterministic drift
remained when standalone scoring and training used different XLA executables: maximum/mean `0.0089493`/`0.00014377`,
ratio mean `0.9999961`, and zero clipping. Backend, attention, accumulation, and standalone `value_and_grad` A/Bs did
not remove it.

Snowball therefore computes old-policy scores with the same compiled accumulated gradient and optimizer program used
by the update. A dynamic flag makes the scoring invocation return the input model, optimizer, RNG, and step state
unchanged while retaining the differentiated-forward score matrix. The following update invokes that same executable
with the independent returned scores and commits its result. Qualification requires bitwise-equal repeated scores,
zero score-versus-training difference, ratio minimum/mean/maximum all exactly `1.0`, and zero clipping. This costs one
discarded backward and optimizer calculation per update; exactness is the chosen gate for the target-size run.

Iris job `/romain/snowball-numerical-exact-mean-01a0a1be` ran the representative one-layer Snowball shape on eight
H100s at MarinSkyRL `049d733b7b416ffd67c273b87383847787196f9b` and the pinned Levanter revision. It succeeded with
no retries or preemptions in 72.333 seconds. Its exact assertions covered a 4096-token batch with two accumulation
steps: repeated score maximum/mean difference `0`, score-versus-differentiated-training maximum/mean difference `0`,
ratio minimum/mean/maximum `1.0`, and all clipping fractions `0`. The immediately preceding run exposed only a
diagnostic artifact: directly reducing an array of unit ratios returned `0.9999999404`. The metric now reduces
`ratio - 1` and adds one after aggregation, preserving exact unit reporting without changing the objective or relaxing
the score gate.

The independent CPU oracle builds the same tiny model in the native PyTorch Grug implementation. It compares selected
response log probabilities, the masked loss, every gradient, and the first AdamW update:

| Quantity | Mean absolute difference | Maximum absolute difference |
| --- | ---: | ---: |
| Token log probability | `1.4305115e-7` | `2.3841858e-7` |
| Masked GRPO loss | — | `4.4703484e-8` |
| Gradient | `5.0938549e-11` | `8.9406967e-8` |
| First AdamW update | `4.0987352e-10` | `1.1920929e-7` |

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
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --frozen \
  --extra cuda --extra vllm --extra levanter-gpu --group dev \
  pytest -s skyrl-train/tests/gpu/levanter_snowball_cycle.py
```

It runs the real `TrajectoryRunner` and public `RayPPOTrainer.train` path. Phase one performs initial publication, two
stochastic rollout/update/publication cycles, exact vLLM readbacks after each final update, and callback checkpoints at
steps 1 and 2. Phase two starts a fresh Ray process, restores step 1 through the public trainer, verifies exact
checkpoint arrays and dataloader position, republishes before generation, replays the saved second batch, and completes
the public second update. The resumed trajectory UIDs and response token IDs exactly match the saved batch.

The current tiny-model gate was Iris job `/romain/snowball-ep2-capstone-idle-fast-01a0a1be`. It passed in 198.96
seconds of test time. The four-H100 pod existed for 227 seconds, or 0.252 allocated H100-hours. Five publications each
verified selected dense values plus expert 0 and expert 4 on opposite EP owners. Ratio minimum, mean, and maximum were
all `1.0` in this exact tiny case, with zero clipping.

| Measurement | Seconds |
| --- | ---: |
| Learner initialization | `4.300` in phase one; `5.064` after restart |
| First forward compilation | `2.298` in phase one; `2.224` after restart |
| First complete update | `9.117` in phase one; `9.421` for fresh-process replay |
| Warm complete update | `0.0343` |
| Warm forward | `0.0227` |
| Warm full iteration | `0.804` |
| Compiled full iteration | `14.410` in phase one; `13.099` after restart |
| Checkpoint commit and MSRL marker | `0.179`–`0.249` |
| Fresh-process checkpoint load | `0.373` |
| Weight publication | `0.309`–`0.634` |

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

Campaign learning quality, sustained learner throughput, a matched Megatron measurement, an exact restore of the
retained 67B S3 checkpoint, and target qualification of the stronger per-expert receipt remain outside the completed
r17 evidence. The tiny reference-kernel timings do not predict 67B throughput.

With the currently resolved `marin-iris` package, Levanter cannot initialize its direct Iris metrics writer because
that package lacks `iris.runtime.telemetry.resolve`. Levanter catches this failure and training continues; MSRL logs and
the returned learner metrics remain available. Direct Levanter telemetry needs a compatible Iris package in a follow-up.

The next expensive checks are a bounded TP1/DP8/EP8 publication using the stronger slice receipt and, when the retained
checkpoint is accessible from a launch environment, a fresh four-host restore followed by one collective operation.
