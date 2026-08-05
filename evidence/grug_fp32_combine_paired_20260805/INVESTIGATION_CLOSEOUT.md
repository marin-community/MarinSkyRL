# Grug FP32 combine investigation: final reviewer packet

Audit date: `2026-08-05` UTC

## Disposition

- MarinSkyRL [#276](https://github.com/marin-community/MarinSkyRL/pull/276)
  remains unchanged at
  `0c213586b5491b8046ca7780e965c4b26dc6a2a2`.
- Evidence-only candidate
  `fbb1fc8378601e0346d00d186809f10d1ad0360d` remains off #276.
- Marin [#7903](https://github.com/marin-community/marin/issues/7903)
  remains open. The distributed action-output residual remains unresolved.
- No experiment, test, benchmark, GPU job, or distributed workload was run for
  this closeout.

The candidate satisfies a bounded local FP32-combine contract. It failed the
unchanged distributed semantic gate and has a measured local time and
combine-memory cost. The evidence does not support putting it on #276.

#276 is candidate-clean and its grouped-expert commits are inspectable, but the
full live PR view is not a clean review or merge surface: it is conflicting
with current `main` and still includes MuonH commits already landed through
#249. This closeout does not change the prohibited product stack. The exact
owner and action are recorded below.

## Exact revisions and durable records

| Item | Identity and durable record |
| --- | --- |
| Live #276 head | `0c213586b5491b8046ca7780e965c4b26dc6a2a2` on `grug-moe-execution-0792437-20260801` |
| Grouped-only review range | [`7276dee1...0c213586`](https://github.com/marin-community/MarinSkyRL/compare/7276dee1f7d9c94d4925bf91a1eff07d0d86295f...0c213586b5491b8046ca7780e965c4b26dc6a2a2), two commits and 11 files |
| Candidate | `fbb1fc8378601e0346d00d186809f10d1ad0360d`; its parent is the live #276 head |
| Candidate preservation | The candidate commit is local/evidence-only. Its exact mail patch and the first candidate-pair harness are in [`7c3bac451a69d34fa8b8f027ceb91998a6e0ff2c`](https://github.com/marin-community/MarinSkyRL/commit/7c3bac451a69d34fa8b8f027ceb91998a6e0ff2c). The archived and local patch IDs are both `222b68f16bb539441edbf7aad1165c0a6df666d4`. |
| Paired local report | [`40d365d661f3acc411a78b8801000a4eae964512`](https://github.com/marin-community/MarinSkyRL/tree/40d365d661f3acc411a78b8801000a4eae964512/evidence/grug_fp32_combine_paired_20260805) |
| Pre-closeout publication audit | [`ec69d90896c8df6c4d74fc53ef91e3ec25a2398b`](https://github.com/marin-community/MarinSkyRL/commit/ec69d90896c8df6c4d74fc53ef91e3ec25a2398b) |
| Later diagnostic source record | [Exact route-aware and causal source record](../grug_divergence_closeout_20260803/later_diagnostics_20260804/README.md) |
| Detailed paired result | [#7903 comment 5186445223](https://github.com/marin-community/marin/issues/7903#issuecomment-5186445223) |
| Product disposition | [#276 comment 5186447968](https://github.com/marin-community/MarinSkyRL/pull/276#issuecomment-5186447968) |

The candidate changes 12 production lines in
`skyrl_train/models/grug_moe.py` and adds one focused CPU regression. The
candidate commit object is not reachable through the public GitHub API, so it
must not be presented as a public commit link. The preserved mail patch is the
durable public source.

## Live #276 review state

Live GitHub readback on `2026-08-05` found #276 open, non-draft, unmerged, and
at the exact head above. Its four-commit stack is:

1. `abad772075631c48a4cb4c3bdcb3145963dc408c`
2. `7276dee1f7d9c94d4925bf91a1eff07d0d86295f`
3. `b37cd1cb4027c0a11d705734554c739e5f9f67f7`
4. `0c213586b5491b8046ca7780e965c4b26dc6a2a2`

The live head is the candidate's parent, and `fbb1fc8` is not an ancestor of
that head. The 16-file PR diff contains no evidence launchers, readers, reports,
or diagnostics. Therefore #276 contains neither the FP32 candidate nor
evidence-only machinery.

The exact-head `lint`, `infra_tests`, `iris_launcher_tests`,
`skyrl_train_tests`, `skyrl_train_macos_install`, and `skyrl_gym_tests`
checks are successful. Two disabled cleanup checks are skipped. There are no
submitted reviews or inline review comments. The review decision is
`REVIEW_REQUIRED`.

There are two independent review-surface limits:

- GitHub reports `CONFLICTING` / `DIRTY` against current `main`.
- The first two #276 commits are also the first two commits of merged
  MarinSkyRL [#249](https://github.com/marin-community/MarinSkyRL/pull/249).
  Because #249 landed through merge commit
  `f1b379be14efdcb831c82a79ce2149b6720ab24a`, the full #276 diff still
  exposes already-landed MuonH implementation, fixtures, and tests.

The two grouped-expert commits remain directly inspectable in the grouped-only
compare linked above. The full PR should not be presented as a clean review or
merge surface until its product owner restacks away the #249 delta and resolves
the current-main conflict. This closeout leaves #276's code, commit stack,
branch, conflict, and checks unchanged.

## What the local evidence proves

Earlier preserved fixed-route localization placed the original eager/grouped
difference at the final BF16 global `index_add_` boundary. The later paired
fixture is independent evidence and did not itself reproduce a parent
eager/grouped output mismatch:

- Parent eager and grouped outputs were bitwise equal to each other. Each
  differed from the independent FP32 reduction in `4,096 / 524,288`
  elements, with maximum error `0.015625`.
- Candidate eager and grouped outputs were bitwise equal to each other and to
  both the independent FP32 reduction and BF16-cast FP64 reduction.
- The five combine-relevant gradient groups passed the frozen tolerance.
  Candidate eager/grouped hidden-gradient maximum difference was
  `2.384185791015625e-7`; the other four targets were bitwise equal.
- Returned BF16 output versus the uncast FP64 reference improved from parent
  maximum/mean error `0.01171875 / 0.000091552734375` to candidate
  `0.00390625 / 0.000030517578125`.

The paired fixture therefore proves that both parent paths share a BF16
accumulation error relative to the independent FP32 reduction and that the
candidate satisfies this local FP32 contract. It does not itself prove removal
of an eager/grouped mismatch, and it does not directly measure the product's
internal accumulator before the final BF16 cast.

The paired real-shape H100 fixture found:

- full fixed-route block forward and backward: `+0.537424` ms,
  `1.038439x`, with all eight GPUs slower;
- complete affected combine forward and backward: `+0.531640` ms,
  `1.340920x`; and
- complete-combine incremental peak allocation: `+251,592,704` bytes,
  `1.461307x`.

This was estimate-only block evidence. It has no predeclared practical
regression threshold and does not measure end-to-end step time or MFU. No
low-level profiler was run, so kernel-level slowdown attribution is not proven.

## What the distributed evidence does not prove

| Evidence | Exact job and artifact | Result and limit |
| --- | --- | --- |
| First `fbb1fc8` 32-H100 pair | Writer `/romain/grug-paired-eager-grouped-fbb1fc8-s1-rno-20260803`; reader `/romain/grug-candidate-headline-readback-fbb1fc8-s1-20260803`; `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-fbb1fc8/headline-paired-s1.json`; payload/result SHA-256 `f06f4d21722e23843758537f72f79c52d3ebb54120b7344ac1f2d622dcdf5546` / `2bf1c1b9ec35ea37adb2180eae51988b49c19e2e34171188f012cdc9219af8e2` | CE and sampled gradients passed; action log probabilities failed `3 / 12,288`. Its `14.227935x` timing ratio is diagnostic only because semantics failed. [Full record](https://github.com/marin-community/marin/issues/7903#issuecomment-5172635389). |
| Route-aware 32-H100 pair | Writer `/romain/grug-route-discriminator-headline-fbb1fc8-s1-rno-cpu8-mem768-20260804`; reader `/romain/grug-route-discriminator-headline-readback-s1-rno-cpu8-mem768-20260804`; `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/headline-discriminator-s1-rno-cpu8-mem768.json`; payload/result SHA-256 `c66485340e2e5d8def34a51fd2fbece3d569de0f6d6277e55f521e0a4fa5247e` / `7393e7cfdcfe284810e62a9fbb6a84a347db259e2ea1fb6cd26d8d58537b1324` | Action log probabilities failed `1 / 12,288` at a new coordinate. The earlier three coordinates did not reproduce. This is not evidence of a stable `3 -> 1` improvement. |
| Focused causal probe | Writer `/romain/grug-causal-probe-headline-fbb1fc8-r22-mb54-l2-s1-rno-cpu8-mem768-20260804`; reader `/romain/grug-causal-probe-headline-readback-fbb1fc8-r22-mb54-l2-s1-rno-20260804`; `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/causal-probe-headline-r22-mb54-l2-s1-rno-cpu8-mem768.json`; payload/result SHA-256 `7904c2d314eff0b0c51d1255a436ad215b809727c9acf63f8b8d298e5273b740` / `b7a4d99efd9057565f7a3be13c5ca1b21d609e12b4c3f08738434c192a8d604b` | `no_causal_internal_divergence` in the captured sparse-block cone. This neither exonerates grouped MoE across runs nor identifies attention as causal. [Full route-aware record](https://github.com/marin-community/marin/issues/7903#issuecomment-5175459688). |
| Local paired H100 cost | `/romain/dev-gpu-romain-grug-fp32-20260805`; raw packet SHA-256 `60e882a908643ae487b6f2f2a1c0c979a6a3d10bb9dccd97c40601c98e02526e` | One `h100-8x` node in `cw-rno2a`, interactive priority; local fixed-route correctness and cost only. [Immutable packet](https://github.com/marin-community/MarinSkyRL/tree/40d365d661f3acc411a78b8801000a4eae964512/evidence/grug_fp32_combine_paired_20260805). |

The latest distributed gate remains failed. These runs do not prove production
equivalence, a grouped-MoE defect, grouped-MoE innocence, attention causality,
or an accepted performance result.

The route-aware and causal launchers, runners, readers, and exact driver/worker
patches are now preserved in the
[later diagnostic source record](../grug_divergence_closeout_20260803/later_diagnostics_20260804/README.md).
Applying each patch to `7c3bac4` reconstructs the writer source hashes pinned
by its launcher. This closes the source-reviewability gap without a new run.

## Levanter contract boundary

Separate checkpoint-provenance work traces the target step-630 recipe to ring
EP8. The historical ring path does not request `fbb1fc8`'s
BF16-product/FP32-local-sum contract, so checkpoint parity is not a reason to
add the candidate. The original training run's exact recorded Git SHA remains
unrecovered, and this closeout does not claim a universal Levanter numerical
contract.

Defining any predeclared ring-specific acceptance contract belongs to the
separate MarinSkyRL/Levanter training-gap owner. That work must not
retroactively validate #7903's failed distributed gate.

## Working-tree isolation

The evidence worktree already contained these unstaged diagnostic files before
the paired measurement closeout:

- modified `skyrl-train/scripts/grug_fixed_replay_benchmark.py`;
- modified
  `skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py`;
- untracked
  `skyrl-train/tests/cpu/workers/test_grug_route_discriminator.py`; and
- untracked `.agents/tmp/*` launch, readback, and review scratch files.

The modified driver and worker are the focused causal-probe source and are now
preserved exactly as a patch in the later diagnostic source record. The
route-aware predecessor was recovered from its pre-causal patch snapshot and
independently reconstructed to the launch-pinned hashes. The CPU test and
remaining scratch files are pre-existing diagnostic work, are not product
changes requested by this closeout, and remain intentionally unstaged.

This closeout neither modifies nor stages those product-tree files. Only the
named evidence documents and source records are included in its commit. The
remaining local diff is therefore explained and isolated rather than silently
mixed into the durable record.

## Final High-tier review and corrections

The required read-only `KIND=GOAL` High-tier review completed before the
GitHub update:

- Claude review SHA-256
  `c123505d86904001f6ab23b7d4a4ed3a7a76d0d0672b949ea3cf3a04bd0cda3e`;
- Codex review SHA-256
  `b4dbd7f4013d3917bf928ac97dcfae820e8234df656704f2ffad83ed2aec191e`;
- Gemini review SHA-256
  `0dd281f5a13d1cacc056352ddc6d759cd5e8a6ace8997fa6c34b485950799607`.

Their material findings were applied: the paired local claim is narrowed to
what its raw packet proves; the duplicated #249 scope and current-main conflict
are assigned to the #276 product owner; the later executed sources are
preserved; the local dirty files are explained and isolated; and the proposed
#7903 update is limited to the genuinely new disposition and ownership facts.

## Unresolved questions and exact next actions

- The non-repeatable distributed residual remains unexplained.
- The original training run's exact recorded Git SHA remains unavailable.
- A predeclared ring-specific action/loss/gradient acceptance contract has not
  been defined.
- Candidate end-to-end step-time or MFU materiality is unmeasured. That matters
  only if a later product decision revives the candidate for a reason other
  than checkpoint parity.

**Immediate owner:** the #276 product owner.

**Immediate action:** restack #276 away from the already-landed #249 delta and
resolve the current-`main` conflict before requesting full PR review or merge.
Use the grouped-only `7276dee1...0c213586` compare for the present code review.

**Separate owner:** the MarinSkyRL/Levanter training-gap owner.

**Separate action:** define a predeclared ring-specific acceptance contract if
checkpoint-parity review still needs one. Keep the non-repeatable #7903
residual open and separate. Do not add `fbb1fc8` for checkpoint parity, and
do not restart the completed distributed residual investigation from this
closeout.
