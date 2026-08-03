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
- Independent reader SHA-256: `e95301b20c4fd028249b6a72a05536d0c7995b269f8cc5e62997d844a53a4221`
- Independent readback launcher SHA-256: `1e8d815c2aa09d3e4a70b97d64f1c408aaaf8c59b7020c5e23845fe920eb02d1`

The readback launcher requires a clean evidence worktree and proves that the
executed source commit is an ancestor of its current HEAD. This permits later
evidence-only commits while the immutable image digest continues to pin the
code that ran.

Iris accepted the exact production-priority request. All four nodes joined one
32-GPU Ray actor group. At `2026-08-03T02:20:35Z`, cgroup memory was 63--67 GiB
per node against the 1,600 GiB limit, and every node reported zero `high`,
`max`, `oom`, `oom_kill`, and `oom_group_kill` events. The job was still
running the expected long eager arm when this entry was written. Final metrics
and independent readback remain pending.
