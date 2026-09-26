# Iceball fixed-input H100 trainer replay

The merged-source replay ran on 2026-09-26 as Iris job
`/atqamar/atqamar-iceball-replay-final-aa4d-20260926` on one non-preemptible
RNO node with eight H100 80 GB GPUs. Its local bundle came from MarinSkyRL
`aa4d824b476ee23c48d1776a1dafce0a059096be`; the historical runtime was
`72cc492d3ba03941715786f558d6be6ae1a52238`. The job and analysis job
`/atqamar/atqamar-iceball-replay-analysis-final-aa4d-20260926` succeeded
with zero failures and preemptions. Raw output has seven-day retention; the
fixed fixture hash, measurement protocol, numerical comparison, and timing
table are retained below.

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
| Historical FSDP2 | 14.8797, 15.1318, 14.7991 | 14.8797 (14.7991–15.1318) | 0.1736 | 376.7 |
| Historical Megatron | 13.5509, 13.4628, 13.4564 | 13.4628 (13.4564–13.5509) | 0.0528 | 417.0 |
| Merged-source Megatron | 14.0604, 14.4646, 14.3569 | 14.3569 (14.0604–14.4646) | 0.2093 | 393.6 |

The historical FSDP2/Megatron median-total ratio was 1.105. The ratio against
the merged-source Megatron runtime was 1.036. Merged-source Megatron's median
was 6.6% longer than historical Megatron's; this compares runtime revisions
and dependencies, not trainer backends. Both Megatron cells had identical
initial, per-update, and final policy probes in every repetition. Timed
intervals sum synchronized maximum-rank `ppo_train` calls;
staging, startup, rollout, probes, checkpoints, export, and evaluation are
outside the timed region.

An earlier matrix job,
`/atqamar/atqamar-iceball-replay-final-bd8a-20260926`, used the same fixture
and harness with current runtime `bd8a9dee466afd3277282321b151bbc9b29da3be`.
Historical FSDP2, historical Megatron, and that runtime's Megatron medians
were 14.5953, 13.4403, and 14.1197 s; its within-job backend ratio was 1.086.
Another independent matrix job,
`/atqamar/atqamar-iceball-replay-matrix-guarded-rno-20260926`, used the same
fixture, harness, and historical source on another RNO H100 node. Its
historical FSDP2 totals were 15.6417, 15.7410, and 15.5528 s; historical
Megatron totals were 13.4708, 13.5637, and 13.2575 s; its earlier current
Megatron bundle totals were 14.2812, 14.1712, and 14.0508 s. The historical
backend ratio was 1.161 in that job. The difference between its FSDP2 times
and the merged-source job's FSDP2 times exceeds the within-job repeat spread.
These measurements support a within-job backend advantage, while node and
launch variation preclude one fixed speedup across jobs. The earlier raw
outputs also have seven-day retention.

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
