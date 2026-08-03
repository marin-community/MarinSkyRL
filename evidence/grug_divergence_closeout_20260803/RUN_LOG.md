# Grug divergence closeout run log

This file records immutable inputs and accepted results for the bounded
divergence closeout. Product code in MarinSkyRL PR #276 is unchanged.

## Eight-H100 confirmatory gate

- Iris job: `/romain/dev-gpu-romain-grug-gate-2dd905e`
- Cluster and allocation: `cw-rno2a`, one node, eight H100s
- Source: `2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2`
- Image: `ghcr.io/marin-community/marinskyrl@sha256:24c655d33ebb6ef78b9f9a5db4053f838c2e9d6c98e3adef338cdb87e1c072a2`
- Artifact: `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-2dd905e/preflight-paired-s1.json`
- Artifact payload SHA-256: `45823262fb0103ff184e27c6d520d701a17ba67511cafad907ff8e1e87767621`
- Canonical result SHA-256: `bb7f4778e49fb7e9586ea9f430436e7cc7a7dda5ed63b0b0a537c6354fa5b914`
- Independent summary SHA-256: `afaa41ec95b1dc17410fb78562132dfec04e1f9ecb8a8365596db07b39bca38d`
- Verdict: pass

The same actor group ran eager oracle, eager instrumented, and grouped
instrumented arms sequentially. The replay, model state, CPU and CUDA RNG,
empty-gradient baseline, worker identity, topology, and rank-to-GPU mapping
were restored before each arm. Every arm returned 476 nonempty gradient
tensors and at least 8,384,693,120 gradient elements per rank, with no
nonfinite gradient tensors.

The eager instrumentation oracle was exact for global CE and representative
action log probabilities. Its representative-gradient maximum allowance ratio
was `0.222545`. Grouped versus eager passed the predeclared global-CE and
action-log-probability checks with maximum allowance ratios `0.441190` and
`0.559959`. Full grouped representative gradients had 72 observational
violations, as the contract anticipated for accepted route discontinuities;
the pinned direct selected-expert output and gradient probes were bitwise
exact.

The route comparison covered 208 rank-layers, 1,666,496 tokens, and 6,665,984
routed allocations. It observed 146,574 changed tokens and 149,971 changed
allocations. All changed memberships satisfied the predeclared adjusted-logit
margin rule; unexplained changed tokens were zero.

The synchronized arm walls were `18.1698891760` seconds for eager oracle,
`18.2898274900` seconds for eager instrumented, and `2.2947125861` seconds for
grouped instrumented. Peak allocated HBM was 63,336 MB for eager instrumented
and 61,400 MB for grouped instrumented. The remote cgroup peaked at
203,363,471,360 bytes and recorded no memory-pressure or OOM event. The gate
allocation was released after successful readback.

## Frozen 32-H100 headline pair

- Iris job: `/romain/grug-paired-eager-grouped-2dd905e-s1-20260803`
- Submitted: `2026-08-03T01:58:54Z`
- Cluster and allocation: `cw-rno2a`, four gang-scheduled nodes, eight H100s
  and 1,600 GiB cgroup RAM per node
- Priority: `production`
- Source and image: identical to the gate above
- Objective: one eager arm followed by one grouped arm on the same actor group,
  matched-CE forward plus backward, no optimizer
- Result URI: `s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-2dd905e/headline-paired-s1.json`
- Remote run script SHA-256: `b939c36ff22cac6f98b48eca09730481d90fd353b8d320fc7ec70fddbca168aa`
- Iris launcher SHA-256: `28f7ca3235a43d6c4174d644034a9cb617da2448cad1567118142e154aae7817`
- Result payload SHA-256: `0f62a8123a280edaf6692ab1db0c01d82b644c8c46906316421f63cc22f2fa8a`
- Canonical result SHA-256: `2c1ef16927846e2ea031077064fd61b84e63bd707b9ec63904169096cb3fbe0c`
- Independent diagnostic reader SHA-256: `ac941919681b7b4c535cdf684b03640473dda79c8e2ae5f9120ade1e370b4d80`
- Independent readback launcher SHA-256: `7f6043650273f54f0e18d0c08ed3cee9f8413e6032909b969da96802d521eb65`
- Independent summary SHA-256: `c9a21098f9529997e534bbe8a1da7b06eee390860c3f6c44e192207ef03dae70`
- Readback job: `/romain/grug-paired-readback-2dd905e-s1-20260803`
- Headline verdict: fail

The readback launcher requires a clean evidence worktree and proves that the
executed source commit is an ancestor of its current HEAD. This permits later
evidence-only commits while the immutable image digest continues to pin the
code that ran.

The pre-run reader (`78b2133facfa1dbe4d37d931a38f1d0496b496639d33c51cf5e6ca8e35279717`)
stopped immediately on a failed semantic verdict. After the uploaded result
failed, the diagnostic reader changed only that early exit into explicit
`semantic_pass=false` and `headline_valid=false` fields so it could continue
checking identity, state, gradients, and metrics. It did not change any
tolerance or acceptance calculation. The launcher added Iris's required
`--enable-extra-resources` declaration after the first CPU submission was
rejected before job creation; no accelerator reran.

Iris accepted the exact production-priority request. All four nodes joined one
32-GPU Ray actor group. The eager and grouped arms both finished, the result
was uploaded, and only then did the driver exit nonzero because the
predeclared headline semantic check failed. Iris consequently marked task 0
failed and its three gang siblings `cosched_failed`. The later Ray zombie and
SIGKILL messages are shutdown cleanup, not the cause.

The independent reader verified the 11,373,214-byte object and its canonical
digest. Initial and final topology and worker identities match within the
recorded actor group. Selection and
warmup restore checks pass. Both arms have 476 finite nonempty gradient tensors
and at least 2,096,173,280 gradient elements per rank. Baseline and finish
gradients are empty, and the final model state is restored. The structural
correctness verdict is therefore pass.

The semantic verdict is fail. Matched global CE differs by only
`0.0000329718` and passes at `0.005470` of its allowance. All 5,184 sampled
representative-gradient checks pass, with maximum allowance ratio `0.810618`.
However, 1,995 of 12,288 representative action log probabilities violate the
predeclared `rtol=4e-2, atol=4e-3` tolerance. Their maximum absolute difference
is `1.955252`, or `30.610567` times the allowance. The headline result is not
semantically valid.

The invalid pair nevertheless records these observational timings:

| Arm | Synchronized wall | Nonpadding tokens/s | Routed expert time | Non-routed time | Peak allocated HBM | Total GPU-s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager, critical rank 17 | 6,866.115339 s | 3,654.966 | 5,675.614517 s | 1,190.500822 s | 37.574300 GiB | 219,715.691 |
| grouped, critical rank 8 | 364.986519 s | 68,757.115 | 75.493324 s | 289.493195 s | 35.763228 GiB | 11,679.569 |

The observed wall and throughput ratio is `18.811970x`, with an observed
208,036.122-GPU-second difference. These are not a causal recovery claim
because the action-output contract failed.

Memory remained safe throughout. Direct samples showed 63--67 GiB cgroup use
per 1,600 GiB task, roughly 1.89 TiB physical RAM available per node, no swap,
and zero memory-pressure or OOM events. Peak reserved HBM was 47.648438 GiB in
both arms.

The routed and non-routed values are each a wall partition on that arm's own
critical rank. They are not a matched cross-rank component attribution.

The eight-H100 gate checked 24 representative action log probabilities. The
headline checked 12,288 under the same frozen rule. The gate's pass therefore
had much less coverage of the full replay distribution; its maximum output
allowance ratio was already `0.559959`.

The exact remaining discriminator is a full-headline, per-token route
membership comparison under the already recorded adjusted-logit margin rule.
The frozen implementation retains every eager microbatch's full adjusted-logit
tensor and selected-expert IDs in device memory until the grouped comparison.
Headline mode has 128 microbatches per rank versus one in the gate, so enabling
the existing route mechanism would scale that retained state by 128x and would
change both HBM headroom and the measured program. The predeclared contract
therefore kept full route adjudication in the gate. This was a bounded harness
choice, and it leaves a real headline evidence gap.

Obtaining that evidence requires changed diagnostics and a second 32-H100 pair.
The next pair is acceptable only if the unchanged output, CE, identity, state,
gradient-finiteness, and oracle rules all pass and route comparison reports zero
unexplained changed tokens, unless a replacement contract is separately
authorized and recorded before new data. Route explanation alone cannot
retroactively validate this pair or its `18.811970x` observation. A second cycle
is beyond the one gate and one pair authorized here; changing the output
tolerance after seeing this result is also forbidden.

Four High reviews found the bounded blocked conclusion correct. Material
provenance and reporting findings were addressed by archiving the exact launch
chain, recording the registry tag-to-digest binding, and running CPU Iris job
`/romain/grug-paired-strict-readback-2dd905e-s1-20260803`. Its verifier asserted
every external pin and recomputed all numeric semantic checks from
raw samples out of process, reproducing the 1,995 output violations and the
failed verdict. The verifier deliberately archives the frozen comparison
formulas. It detects inconsistent embedded records, pin drift, and artifact
corruption, but it would not detect a formula defect shared with the driver.
See `VALIDATION_AND_REVIEW.md`. MarinSkyRL #276 remains unchanged.

## Publication

The bounded result was published at
https://github.com/marin-community/marin/issues/7903#issuecomment-5162561004.
Readback confirmed that #7903 remains open. MarinSkyRL #276 remained unchanged
at `0c213586b5491b8046ca7780e965c4b26dc6a2a2`, open, non-draft, clean,
mergeable, green, and unmerged. No #276 comment was added because this pass
demonstrated no permanent grouped-runtime defect and did not change its product
classification.
