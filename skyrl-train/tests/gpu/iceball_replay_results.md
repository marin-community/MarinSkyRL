# Iceball fixed-input H100 trainer replay

The bd8a-revision replay ran on 2026-09-26 as Iris job
`/atqamar/atqamar-iceball-replay-final-bd8a-20260926` on one non-preemptible
RNO node with eight H100 80 GB GPUs. It used local-bundle MarinSkyRL source
`bd8a9dee466afd3277282321b151bbc9b29da3be` and historical source
`72cc492d3ba03941715786f558d6be6ae1a52238`. The job and its analysis
job `/atqamar/atqamar-iceball-replay-analysis-final-bd8a-20260926` succeeded
with zero failures and preemptions. The [raw output archive](s3://marin-us-east-02a/tmp/ttl=7d/iris/task-outputs/atqamar/atqamar-iceball-replay-final-bd8a-20260926/0/4f7db232d6ef9093/outputs.tar.zst)
has seven-day retention; the fixed fixture hash, measurement protocol,
numerical comparison, and timing table are retained below.

Run `iceball_replay_matrix.sh` in the frozen CUDA runtime with
`--model-uri s3://marin-us-east-02a/marin/checkpoints/iceball-micro-sft/2026.09.26/hf/step-7`,
`--model-identity checkpoints/iceball-micro-sft@2026.09.26:b7c148b4`,
`--cells old-fsdp2 old-megatron new-megatron`, `--repetitions 3`, and
`--measured-steps 8`. The matrix script fetches historical MarinSkyRL commit
`72cc492d3ba03941715786f558d6be6ae1a52238`, installs its frozen FSDP2 and
Megatron environment, and copies the same replay harness into that checkout.
The harness SHA-256 was
`442db32dbb7c4466e56db37a63a556efd4d792c99cbe4fdb500c0d1822cd82e8`.

Each cell staged identical SFT weights and ran three repetitions. Every
repetition used eight ordered batches of 64 sampled sequences (5,626 valid
response tokens) and eight optimizer updates. The fixed tensors hashed to
`7717d528cc8feda8045681e8ae4ea446202724fda2cd9e12e4efdfe0c3605a4f`.
All cells used DP8, microbatch one, BF16 parameters, AdamW at 2e-6, and gradient
clipping at 1.0. Two untimed warmup updates preceded a fresh optimizer and
worker state per repetition. The cells alternated backend order on the same
node. The predeclared maximum absolute starting-logprob guard was 0.05; all
FSDP2 repetitions passed, with an observed maximum of 0.01563.

| Trainer | Eight-update totals (s) | Median total (range, s) | Sample standard deviation (s) | Mean valid tokens/s |
| --- | --- | --- | --- | --- |
| Historical FSDP2 | 14.6441, 14.5953, 14.5415 | 14.5953 (14.5415–14.6441) | 0.0513 | 385.5 |
| Historical Megatron | 13.4403, 13.5245, 13.3830 | 13.4403 (13.3830–13.5245) | 0.0712 | 418.3 |
| Current Megatron | 14.0007, 14.1197, 14.3674 | 14.1197 (14.0007–14.3674) | 0.1871 | 397.2 |

The historical FSDP2/Megatron median-total ratio was 1.086. The ratio against
the current Megatron runtime was 1.034. The current Megatron median was 5.1%
slower than historical Megatron; this compares runtime revisions and
dependencies, not trainer backends. Historical and current Megatron had
identical policy probes, policy losses, and gradient-norm receipts at every
update. Timed intervals sum synchronized maximum-rank `ppo_train` calls;
staging, startup, rollout, probes, checkpoints, export, and evaluation are
outside the timed region.

An earlier independent matrix job,
`/atqamar/atqamar-iceball-replay-matrix-guarded-rno-20260926`, used the same
fixture, harness, and historical source on another RNO H100 node. Its
historical FSDP2 totals were 15.6417, 15.7410, and 15.5528 s; historical
Megatron totals were 13.4708, 13.5637, and 13.2575 s; its earlier current
Megatron bundle totals were 14.2812, 14.1712, and 14.0508 s. The historical
backend ratio was 1.161 in that job. The difference between its FSDP2 times
and the final job's FSDP2 times exceeds the within-job repeat spread, so these
measurements support a modest within-job backend advantage but not one fixed
speedup across nodes and launches. The earlier [raw archive](s3://marin-us-east-02a/tmp/ttl=7d/iris/task-outputs/atqamar/atqamar-iceball-replay-matrix-guarded-rno-20260926/0/85f3b4b40a99be0a/outputs.tar.zst)
has the full receipts.

Historical FSDP2 and Megatron starting response-token logprobs differed by
mean absolute 0.000164 and maximum absolute 0.01559. After eight updates,
their mean absolute difference was 0.004928 and maximum absolute difference
was 0.016849. Across rank-0 updates, maximum policy-loss difference was
0.000223 and maximum raw-gradient-norm difference was 0.002637. Both trainers
applied eight finite, nonzero policy updates. FSDP2 writes BF16 optimizer
parameters; Megatron maintains FP32 master parameters. Different attention
kernels can also affect the trajectory. This replay does not isolate those
causes. A repeated FSDP2 FP32-master cell failed the starting-weight guard and
is excluded from the matched comparison.
