# FA4 latest-first Grug experiment: results

Status: experimental branch only; FA2 remains the default. The recommendation
is to **keep FA2 and defer FA4 for Grug**. FA4 beta29 can run Megatron on both
H100 and GB200, but the fixed Marin vLLM wheel and FA4 require incompatible
TVM FFI versions at runtime. The newer beta30/31 also silently lost
full-context gradients, the representative Grug geometry did not pass its
existing HF parity gate in either matched arm, and no policy-step speedup was
resolved from observed variation. This is not a production qualification of
beta29 or of the new Megatron cohort.

## Exact cohort and artifact provenance

| Component | Old Marin FA2 reference (`4d798b12`) | New FA2 arm | New FA4 arm |
| --- | --- | --- | --- |
| Torch / CUDA | 2.13.0 / 13.2 | same | same |
| Marin vLLM | `0.0.0.dev20260916+marin.70ea9ae8f260.cu132` | same | same |
| Megatron Core / Bridge | 0.18.0 / 0.6.0 | 0.19.2 / 0.6.2 | same |
| Transformer Engine | 2.11.0 | 2.19.0 | 2.19.0 |
| Attention package and explicit selector | FA2 2.8.3, `NVTE_FLASH_ATTN_V4=0` | FA2 2.8.3, `NVTE_FLASH_ATTN_V4=0` | `flash-attn-4[cu13]==4.0.0b29`, `NVTE_FLASH_ATTN_V4=1` |
| Other coupled packages | ModelOpt 0.46.1, Quack 0.6.4 | same, plus CUTLASS DSL 4.6.2, Hydra 1.3.4, cuDNN Frontend 1.29.0 and TVM FFI 0.1.14.post0 overrides | same |

The new TE wheel was built from official source commit
`5e52befd5262c06289106338c308079d6adb391f` against the fixed Torch/CUDA
line. Its [x86_64 and arm64 prerelease assets](https://github.com/marin-community/MarinSkyRL/releases/tag/fa4-te219-cu132-20260920-694f3adf)
were downloaded and checked against staged SHA-256 values
`1eb84026d9617aed8c656d91877d19ff70647697d6a2993de90063e6a6c37d56`
and `185d4dd78a26623351d7ea22ac3a353d6c675bdee47f19b6d8839618473c82ac`.
The retained [FA4 beta29 release wheel](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta29)
has SHA-256 `85dd4b0e98ca38f5681a3214b2f9f230638030e3b868a50f3873336fc98d7537`.
Marin's [FA2 wheels](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-fa283-20260920)
are available for both x86_64 and SM100 arm64. The stock FA2 and FA4 wheel
paths do not overlap, so a combined wheel was unnecessary. The baseline
Megatron extra is x86_64-only; the three-arm comparison is therefore H100,
while GB200 compares the two new-cohort arms.

**The lock is not a valid rollout environment.** The fixed Marin vLLM wheel's
metadata pins `apache-tvm-ffi==0.1.11` and `tilelang==0.1.12`; FA4 beta28 and
beta29 metadata require `apache-tvm-ffi>=0.1.12,<0.2`. The prototype override
selected `0.1.14.post0` to resolve, but the
[four-H100 rollout/train/broadcast/rollout test](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-grug-rollout-h100-9532de51)
failed before training: a vLLM engine aborted with native `tvm::ffi::Error`
(`TypeAttr __ffi_repr__ is already registered`) when TileLang loaded its
bundled TVM. SkyRL reported repeated `EADDRINUSE`/engine-init retries, but
that was a misleading wrapper, not the first causal failure. A disposable
x86_64 CPU import probe using the same TileLang version reproduced the abort
with TVM FFI `0.1.12` and `0.1.14.post0`; it succeeded with vLLM's pinned
`0.1.11`. No mixed-version override is retained as a working serving path.
Changing the fixed vLLM wheel or its runtime behavior is outside this goal;
there is therefore no valid end-to-end RL timing or rollout qualification.
This runtime incompatibility is independent of the Megatron FA2/FA4 kernel
comparisons below, but blocks a production migration even if they improved.

The retained lock deliberately uses TE 2.19 rather than Core/Bridge's tracked
2.18: this is a compatibility experiment. `uv lock --check --offline`,
architecture-specific sync dry-runs, and frozen-lock imports on
[H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-full-lock-import-3ef87994)
and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-arm-full-lock-import-9f284477)
passed. The GB200 import was with the earlier beta31 lock; the beta29 change
does not alter its Megatron/TE wheel closure. These are installation gates,
not workload validation.

## Backend, gradient, and numerical evidence

All GPU comparisons ran with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2`,
`NVTE_FUSED_ATTN=0`, `NVTE_FLASH_ATTN=1`, and explicit
`NVTE_FLASH_ATTN_V4=0/1`. Transformer Engine's `Selected backend` logs in
the linked jobs show FA2 or FA4 as requested; version imports alone were not
counted. The initial [H100 backend probe](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-h100-backend2-694f3adf)
and [GB200 backend probe](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-gb200-backend2-e99529c2)
both completed finite forward/backward with beta31 on a short local-window
shape. That short pass missed the full-causal failure below.

Beta31 produced **zero pre-optimizer fused-QKV and attention-gate gradients**
in Grug's full-context layers 3 and 7 on both
[H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-grad-h100-b8492530)
and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-grad-gb200-b8492530),
where FA2 had thousands of nonzero entries. Beta30 reproduced this on
[GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-b30-gb200-44dcfe90);
beta29 restored nonzero gradients and moving Q weights on
[H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-b29-h100-44dcfe90)
and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-b29-gb200-44dcfe90).
An independent FP32 math-SDPA reference on H100 found FA2 output mean error
`0.000577` versus beta31 `0.3491` on short full-causal attention
([job](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-attn-sdpa-h100-89bdeb9a)).
With beta29, FA2 and FA4 both had mean output error `0.000577` and mean Q
gradient error about `5.24e-8` on both
[H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-sdpa-h100-9532de51)
and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-sdpa-gb200-9532de51).
The [upstream beta30 change](https://github.com/Dao-AILab/flash-attention/pull/2490)
to causal-window normalization is a plausible cause of the version boundary,
not a proven root cause or a fix in this branch.

On the PP1 toy Grug test (hidden 64, 12-token prompt, 8-token response,
five PPO updates), the [H100 three-arm run](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-grug-toy5-h100-9532de51)
showed the old and new FA2 arms' initial and post-update log-probs and sampled
weight changes were exact. This isolates the kernel comparison in that small
test. Beta29 FA4 and new FA2 had exact initial valid-token log-probs; after
five updates, FA4-versus-FA2 max/mean absolute difference was
`0.03118/0.00403`. On the [GB200 new-cohort pair](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-grug-toy5-gb200-9532de51),
initial log-probs were also exact and post-five max/mean was
`0.02468/0.00458`. All values were finite and sampled attention weights
changed. These are small, high-learning-rate tests, not tolerance changes or
proof of long-run numerical equivalence.

The Snowball-like diagnostic uses hidden 2560, 20 query heads, 5 KV heads,
head dimension 128, 4 layers, 256 experts, window 2048, and variable-length
2400+300-token rows. Its existing strict HF parity gate fails **before any
update for FA2 as well as FA4**: FA2 max/mean error `0.6221/0.0472` on H100
and `0.8261/0.0468` on GB200. The old H100 FA2 baseline has the same
`0.6221/0.0472` result, so this is not a new-refresh regression.
`--diagnose-parity-failure` records matched data and exits nonzero; these runs
are failures, not passes. In the [H100 five-step diagnostic](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-snowball-diag-h100-9532de51),
FA4-versus-FA2 pre-update log-probs differed max/mean `0.8532/0.0402`, and
after five updates `43.35/17.97`; [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-snowball-diag-gb200-9532de51)
was `0.8065/0.0443` before and `40.66/12.62` afterward. Beta29's sampled
Q and gate weights did move and remained finite. Even after only one update,
the diagnostic FA4-versus-FA2 max/mean difference was `2.196/0.423` on
[H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-snowball-one-h100-9532de51)
and `2.054/0.475` on
[GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-snowball-one-gb200-9532de51).
The short test uses learning rate 0.02, so the update divergence cannot by
itself assign the cause to FA4. It does block a numerical equivalence or
throughput claim for this representative geometry.

## Timing and topology limits

The 2700-token causal/sliding-window GQA attention microprobe ran ten steady
forward/backward samples per arm in separate processes. On
[H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-attn-2700-h100-9532de51),
FA2 median total was `2.784 ms` (mean `2.783`, SD `0.011`) versus beta29 FA4
`1.842 ms` (mean `1.844`, SD `0.010`). On
[GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-attn-2700-gb200-9532de51),
it was `2.511 ms` (mean `2.509`, SD `0.023`) versus `1.241 ms` (mean `1.250`,
SD `0.042`). Peak allocated memory was about `618 MB` FA2 versus `585 MB`
FA4 in this probe. Outputs and Q/K/V gradients were finite and close; H100
output max/mean difference was `0.0078125/0.0000202`, GB200
`0.00390625/0.0000763`. These are kernel results, not RL throughput.

Excluding the first JIT/initialization step, toy PP1 Grug policy-update means
over four subsequent steps were H100 new FA2 `0.478 s` (sample SD `0.002`),
FA4 `0.486 s` (SD `0.034`); GB200 new FA2 `0.732 s` (SD `0.010`), FA4
`0.759 s` (SD `0.039`). Process order and cluster load were
not randomized. The single-run differences are within observed variation and
do not establish a policy-step speedup. The Snowball-like five-step runs also
show no clear step-time win but fail parity and are diagnostic only.

The CP2 probe must enable sample packing and `cp_comm_type=all_gather` to use
sliding-window attention with TE 2.19. Real Grug currently disables packing,
so this is **not** its production configuration. With beta29, the refreshed
FA2 arm had a NaN grad norm and nonfinite sampled attention weights after one
update on [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa2-9532de51)
and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa2-gb200-9532de51).
The matched FA4 arm completed with finite gradients and moving sampled weights
on [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa4-9532de51)
and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa4-gb200-9532de51).
The old TE 2.11 baseline cannot run this CP2 toy geometry with either default
`p2p` sliding-window transport or `all_gather` packed causal masking
([p2p](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-cp2-baseline-default-9f284477),
[all_gather](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-cp2-baseline3-c3913161)).
`a2a` cannot use this toy's one KV head or Snowball-like's five KV heads with
CP2. Because there is no valid three-arm CP2 reference and the new FA2 arm
fails numerically, CP2 is a documented experimental result, not a migration
gate passed.

## Reproduce and interpret

Use this branch's frozen `uv.lock` and the scripts in this directory. For
example, on H100 run `bash experiments/fa4/run_three_arm_h100.sh 1 toy 5`;
on GB200 run `bash experiments/fa4/run_grug_h100.sh 1 toy 5` (despite the
historical script name). For attention use
`bash experiments/fa4/run_attention_pair.sh 4 2700 20 5 128 2048 3 10`; for a failed-parity diagnostic
use `bash experiments/fa4/run_diagnostic_pair.sh 1 snowball 1`. Iris job links
above preserve command, GPU type, logs, and terminal result. Do not treat
environment imports, one-step JIT timing, the failed Snowball parity runs, or
the experimental packed CP2 path as production evidence. No production PR is
part of this experiment.
