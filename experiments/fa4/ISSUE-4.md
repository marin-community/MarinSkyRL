## Current status (2026-09-20)

Marin now uses **Torch 2.13 / CUDA 13.2 and a compiled FlashAttention 2.8.3
wheel** for its Megatron path. The old Torch 2.11/cu130, stub-package,
eager-attention, and vLLM 0.20.2 account is obsolete after #561 and
subsequent dependency work. This issue now tracks whether to promote FA4 for
Grug training, not how to make flash attention install at all.

**Recommendation: keep FA2 and defer FA4.** The latest-first prototype found a
working FA4 beta29 Megatron kernel path, but no supported production migration
result. FA4's TVM FFI dependency conflicts at runtime with the fixed Marin
vLLM wheel, and the four-GPU rollout test aborts before training. Beta30/31
also silently zeroed full-context Grug attention gradients; beta29 did not
improve observed toy Grug policy-step time; and the representative Snowball-like
forward failed the existing HF parity gate for both new-cohort arms. We did
not open a production PR or alter the FA2 default.

### Exact experimental matrix

| | Old Marin FA2 | Refreshed FA2 | Refreshed FA4 |
| --- | --- | --- | --- |
| Reference | `4d798b12` | **[prototype code/lock commit](https://github.com/marin-community/MarinSkyRL/commit/9532de511531e1cc55990a1b08fa974f49807544)** | same |
| Torch / CUDA / vLLM | 2.13.0 / 13.2 / `0.0.0.dev20260916+marin.70ea9ae8f260.cu132` | same | same |
| Megatron Core / Bridge / TE | 0.18.0 / 0.6.0 / 2.11.0 | 0.19.2 / 0.6.2 / 2.19.0 | same |
| Attention | FA2 2.8.3 | FA2 2.8.3 | FA4 `4.0.0b29` |
| Explicit TE selector | `NVTE_FLASH_ATTN_V4=0` | `=0` | `=1` |
| ModelOpt / Quack / CUTLASS DSL / TileLang | 0.46.1 / 0.6.4 / 4.6.2 / 0.1.12 | same | same |
| Hydra / cuDNN Frontend / TVM FFI | 1.3.2 / 1.26.0 / 0.1.11 | 1.3.4 / 1.29.0 / 0.1.14.post0 | same |

FA2 remains the selected attention default; FA4 is an opt-in `--extra fa4`.
But the experimental TVM FFI override is global: **this retained lock cannot
serve with the fixed vLLM wheel even when FA2 is selected**. The Megatron/TE
refresh without FA4 and without that override was not rollout-tested, so its
serving viability remains unknown. Core/Bridge track TE 2.18, so TE 2.19 here is an *unqualified compatibility
experiment*. The fixed Marin vLLM wheel and Torch/CUDA line did not change.
The [TE 2.19 experimental x86/arm64 wheels](https://github.com/marin-community/MarinSkyRL/releases/tag/fa4-te219-cu132-20260920-694f3adf)
were built from source `5e52befd5262c06289106338c308079d6adb391f`;
hashes, lock, reproduction commands, and full job matrix are in the
[prototype results](https://github.com/marin-community/MarinSkyRL/blob/71b3635ab50737e28fb8a8c451929c1c83b2db01/experiments/fa4/RESULTS.md).
The old Megatron extra is x86-only, so the three-arm comparison is H100;
GB200 compares the two refreshed arms.

### What ran

- Frozen-lock imports passed on [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-full-lock-import-3ef87994)
  and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-arm-full-lock-import-9f284477).
  TE logs explicitly selected the requested FA2/FA4 backend in the linked GPU
  jobs; importing FA4 was never treated as backend proof.
- **End-to-end serving is blocked.** The fixed Marin vLLM wheel pins
  `apache-tvm-ffi==0.1.11` and `tilelang==0.1.12`; FA4 beta28/29 require
  `apache-tvm-ffi>=0.1.12,<0.2`. The experimental lock overrides to
  `0.1.14.post0` for every Linux install, but the
  [four-H100 rollout test](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-grug-rollout-h100-9532de51)
  aborts in vLLM startup with `tvm::ffi::Error: TypeAttr __ffi_repr__ is
  already registered` when TileLang loads its bundled TVM. SkyRL's repeated
  “port collision” retries are a misleading wrapper. A disposable import
  probe reproduced the native abort with FFI 0.1.12 and 0.1.14.post0, but
  passed with vLLM's pinned 0.1.11. We stopped this dependent experiment;
  changing the fixed vLLM wheel is outside this pass. There is **no valid
  end-to-end RL timing or rollout pass**.
- On [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-grad-h100-b8492530)
  and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-grad-gb200-b8492530),
  latest beta31 gave **zero fused-QKV and attention-gate gradients** in
  full-context Grug layers where FA2 had thousands of nonzero elements.
  [Beta30](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-b30-gb200-44dcfe90)
  reproduced it; [beta29 H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-b29-h100-44dcfe90)
  and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-b29-gb200-44dcfe90)
  restored moving attention weights. An independent FP32 SDPA comparison also
  found beta31's H100 full-causal output mean error `0.3491` versus FA2
  `0.000577`; beta29 matched FA2's error on
  [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-sdpa-h100-9532de51)
  and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-sdpa-gb200-9532de51).
- The [H100 three-arm toy Grug run](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-grug-toy5-h100-9532de51)
  completed forward, backward, and five PPO updates. Old and refreshed FA2
  matched exactly; beta29 FA4 matched initial valid-token log-probs and
  differed after five updates by max/mean `0.0312/0.0040`. The
  [GB200 refreshed pair](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-grug-toy5-gb200-9532de51)
  also matched initially and differed after five updates by
  `0.0247/0.0046`. All sampled values were finite.
- On a 2700-token attention-only forward/backward, ten steady samples gave
  H100 median `2.784 ms` FA2 versus `1.842 ms` FA4
  ([job](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-attn-2700-h100-9532de51));
  GB200 `2.511 ms` versus `1.241 ms`
  ([job](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-attn-2700-gb200-9532de51)).
  Peak allocated memory in that probe was about 618 versus 585 MB.
  Four post-JIT toy Grug policy steps averaged H100 old FA2 `0.548 s`
  (sample SD `0.020`), refreshed FA2 `0.478 s` (SD `0.002`), and FA4
  `0.486 s` (SD `0.034`); GB200 refreshed FA2 `0.732 s` (SD `0.010`) versus
  FA4 `0.759 s` (SD `0.039`). The observed old-to-new FA2 difference is
  about 12.7% in this single fixed-order toy run. Process order, separate
  environments, and load were not controlled, so it is not a qualified
  dependency-refresh throughput gain. These runs show **no resolved FA4
  policy-step win**, not a significant slowdown.
- Snowball-like hidden-2560, 2400+300-token Grug is a failed diagnostic gate.
  Both FA2 and FA4 fail the existing strict HF parity check before an update;
  old H100 FA2 fails identically, so this is not a new-refresh regression.
  Matched FA4-vs-FA2 pre-update max/mean log-prob differences were
  `0.853/0.040` on [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-snowball-diag-h100-9532de51)
  and `0.807/0.044` on [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-snowball-diag-gb200-9532de51).
  One-update diagnostics are recorded in the [results](https://github.com/marin-community/MarinSkyRL/blob/71b3635ab50737e28fb8a8c451929c1c83b2db01/experiments/fa4/RESULTS.md).
  Neither failed-parity timing nor five high-LR steps are production
  throughput evidence.
- CP2 is only an **experimental packed/all-gather** path; active Grug disables
  packing. Under beta29, refreshed FA2 produced a NaN grad norm and nonfinite
  sampled weights on [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa2-9532de51)
  and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa2-gb200-9532de51),
  while FA4 completed with finite values on
  [H100](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa4-9532de51)
  and [GB200](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b29-cp2-fa4-gb200-9532de51).
  Old TE 2.11 cannot run this toy CP2 mask/window geometry with the tested
  transports. Thus CP2 has no valid three-arm reference and does not qualify
  a migration.

### Revisit criteria

Reconsider FA4 only after a proposed candidate release passes the full-causal
gradient check against an independent attention reference; a supported
FA4/TVM FFI/vLLM dependency combination runs; representative Grug HF parity
and a usable matched CP topology pass without tolerance relaxation; and repeated,
order-controlled *end-to-end* RL measurements show a meaningful benefit on
the target hardware. Until then, retain FA2 and the current production
dependency cohort. The older issue proposal to remove FA2 and switch the
nightly wholesale to FA4 is withdrawn.

The smallest next dependency options are an FA4 release compatible with
vLLM's pinned FFI 0.1.11, or a separately authorized and qualified vLLM/TileLang
wheel that tolerates FA4's newer FFI. Neither is part of this prototype.
