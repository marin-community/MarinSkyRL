# Iceball fixed-input H100 trainer replay

The replay ran on 2026-09-26 as Iris job
`/atqamar/atqamar-iceball-replay-matrix-guarded-rno-20260926` on one
non-preemptible node with eight H100 80 GB GPUs. It succeeded with zero failures
and preemptions. Raw task output had seven-day retention; the fixed fixture
hash, measurement protocol, numerical comparison, and timing table are retained
below. The analysis job was
`/atqamar/atqamar-iceball-replay-analysis-detail-20260926`.

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

| Trainer | Eight-update totals (s) | Median total (range, s) | Median step (s) | Mean valid tokens/s |
| --- | --- | --- | --- | --- |
| Historical FSDP2 | 15.6417, 15.7410, 15.5528 | 15.6417 (15.5528–15.7410) | 1.8973 | 359.6 |
| Historical Megatron | 13.4708, 13.5637, 13.2575 | 13.4708 (13.2575–13.5637) | 1.1724 | 418.9 |
| Current Megatron | 14.2812, 14.1712, 14.0508 | 14.1712 (14.0508–14.2812) | 1.1966 | 397.1 |

The historical FSDP2/Megatron median-total ratio was 1.161. The ratio against
the current Megatron runtime was 1.104. The current Megatron median was 5.2%
slower than historical Megatron; this compares runtime revisions and
dependencies, not trainer backends. Historical and current Megatron had
identical policy probes, policy losses, and gradient-norm receipts at every
update. Timed intervals sum synchronized maximum-rank `ppo_train` calls;
staging, startup, rollout, probes, checkpoints, export, and evaluation are
outside the timed region.

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
