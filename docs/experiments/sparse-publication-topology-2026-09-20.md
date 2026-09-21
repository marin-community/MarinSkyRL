# Sparse weight publication across topologies (2026-09-20)

The corrected fully asynchronous Grug comparison sent **94.4% fewer routed-expert bytes** with GPU sparse indices. Expert-block installation improved by only **5 ms median**, while the full weight-sync interval was **46 ms slower median**. The sparse arm's 20 post-warmup steps took 3.5% less wall time but accepted 5.2% fewer response tokens; accepted response tokens per step-wall second were 1.7% lower. With one run per arm and different async rollout mixes, there is **no demonstrated job-throughput gain** attributable to sparse publication. Keep dense as the default; offer GPU sparse routed experts as a run-level opt-in where fabric bytes or contention matter and GPU memory can hold the prior image. No production cross-region publisher has been validated.

| Topology | Decision | Evidence and missing measurement |
| --- | --- | --- |
| GPUs in one NVLink domain | Use dense pending a matched trainer run in that domain. | [#698](https://github.com/marin-community/MarinSkyRL/issues/698) found dense local copy faster than sparse encodings on one H100. That was not an inter-GPU NVLink test. The fitted NVLink envelope below is extrapolation, not a measured crossover. |
| Separate H100 nodes in one datacenter | Dense default; GPU sparse routed experts are a controlled opt-in for byte pressure. | The corrected 40-H100 job measured 130.862 versus 7.343 GB global expert payload and 13,312 versus 7,956 broadcast collectives per update, but no clear useful-throughput improvement. The cluster requests InfiniBand devices and uses NCCL; the **selected NCCL network plugin/interface was not recovered** from retained logs, so the physical transport must be verified before generalizing to another fabric. |
| Cross-region trainer and inference | No operational dense or sparse remote publisher is qualified here. Use a dense object as the control for a bounded remote prototype, then test CPU sparse on the actual backend. | The retained one-object H100→S3→GB200 relay and new one-H100 CPU staging probe provide component costs, not full-model async behavior. No equivalent-topology NIXL comparison exists. |

## Controlled runs and exactness

The exactness qualification `/romain/sparse-topology-qualification-v3-20260920` used runtime `870f4cdfdec1b4f72cba9261a482926d81897060`, `max_staleness_steps=0`, and independent dense replay (`verify=true`). On the real five-node Grug topology, the initial dense seed and four sparse publications compared every receiver's raw BF16 bytes, including DP replicas, against an independently formed dense reference; expected schedule bytes matched and the job succeeded. The four-H100 gate also detected deliberate corruption. Replay timing is excluded from performance analysis.

A High-tier GOAL review found that the first 25-step pair, runtime `870f4cdf…`, had `generator.max_num_seqs=32` active on all eight vLLM engines and `offload_optimizer_during_rollouts=true`. It is a valid **capped, offloaded pilot**, but it does not represent the cited production-like [#8955 E9.2 operating point](https://github.com/marin-community/marin/issues/8955#issuecomment-5663169886). Its 20 post-warmup steps had nearly equal wall sums (1190.78 s dense, 1188.70 s sparse), 130.862 versus 6.811 GB encoded expert bytes, and 3.987 versus 4.199 s median full sync. These negative/diagnostic results remain in the durable raw artifacts and are not used to decide production-like throughput or calibrate the final direct model.

The corrected recipe is `cloud/iris/configs/sparse_publication_topology_production.yaml` at runtime `d7598efccec73c62131fbe1c6164a9a82e294d11`. It retains model `6808fe5c219471517bd51df35addefd38ebebf89`, seed 17, the same GSM8K S3 data, 25 updates, maximum staleness 4, one update per batch, 32 buffered groups, 160 generation workers, four responses per prompt, 8192/4096 context/output limits, four 8×H100 trainer nodes (PP2/EP8/DP2) and one 8×H100 vLLM node (DP8/EP8) on `cw-rno2a`. It uses `offload_optimizer_during_rollouts=false`, `gpu_memory_utilization=0.75`, `max_num_seqs=1024`, `enforce_eager=false`, and replay disabled. The separate one-step sparse preflight succeeded on all five nodes before these arms. Resolved configs for the final dense and sparse arms differ only in run-specific paths/names and the sparse arm's `generator.expert_block_sync.encoding=sparse_index` override. The source and model revisions are identical. Source-specific parser/mask settings differ from #8955; this is a Grug publication experiment, not an RL quality replication.

Both final jobs succeeded naturally with zero failures and preemptions. The table uses steps 6–25 (20 updates per arm), excluding startup steps 1–5. Parentheses give the inclusive interquartile range.

| Post-warmup metric | Dense | GPU sparse |
| --- | ---: | ---: |
| Full `sync_weights` | 0.559 s (0.545–0.585) | 0.606 s (0.582–0.630) |
| Expert-block install | 0.441 s (0.441–0.442) | 0.436 s (0.433–0.442) |
| Expert phase | 0.347 s | 0.348 s |
| Sender baseline commit, after acknowledged resume | — | 0.0213 s driver, 0.0108 s rank max |
| Global logical expert / encoded bytes | 130.862 / 130.862 GB | 130.862 / 7.343 GB (6.485–8.074) |
| Changed expert density | — | 1.87% (1.65–2.06%) |
| Global expert broadcast collectives | 13,312 | 7,956 |
| Sender PyTorch GPU allocation peak | 45.827 GB | 54.029 GB (+8.202 GB; prior image 8.179 GB/root) |
| Short-lived GPU peak above pre-publication allocation | — | 30.34 MB sender, 13.63 MB receiver |
| Sender cumulative process peak RSS (`ru_maxrss`) | 12.157 GB | 12.166 GB |
| Step duration | 17.380 s (14.387–18.779) | 17.410 s (9.490–23.394) |
| Sum of 20 step durations | 366.94 s | 353.99 s (−3.53%) |
| Accepted response tokens in 20 steps | 5.104 M | 4.838 M (−5.21%) |
| Accepted response tokens / summed step time | 13,909/s | 13,668/s (−1.74%) |
| Accepted response tokens / summed training interval | 30,754/s | 29,899/s |
| Generation-buffer wait | 8.212 s | 8.204 s |
| Training interval | 8.172 s | 8.071 s |
| Mean rollout staleness | 3.406 | 3.313 (maximum 4 in both) |
| Stale attempts rejected and retried, sum | 60 | 67 |
| vLLM median running requests, per engine | 33.94 | 33.38 |
| vLLM median waiting requests, per engine | 0 | 0 |
| vLLM median of global peak running requests | 373.5 | 355.5 |
| Logged `vllm/median_generation_throughput` | 2,162 | 2,024 |

The accepted-token rates use `sum(generate/avg_num_tokens × 128)` divided by summed step time or summed `timing/run_training`. They exclude prompts and padding and are not exact total-training-token throughput. Async arrival order changed accepted work despite the same seed, so neither the 3.5% wall-sum difference nor the 1.7% token-rate difference isolates transport. There is one run per arm, and within-run IQR does not measure run-to-run variation. Unlike the pilot, the 32-sequence cap is absent: global peak running requests exceeded 256 on 20/20 dense and 19/20 sparse post-warmup updates, median waiting requests were zero, and GPU KV cache usage stayed low. The expert install benefit was too small to produce a resolvable throughput gain at this operating point.

With optimizer offload disabled, the full sync is about 0.56–0.61 s. The pause-call timer measures only the call to pause (0.097 s dense, 0.127 s sparse median), not the whole pause-to-resume window; subtracting commit from full sync gives about 0.559 versus 0.583 s median as a **rough** pause-window estimate. The full interval also includes drain and resume. Sparse commit occurs after acknowledged resume and adds 21 ms median to the broad interval. GPU component timers (`detect`, `pack`, `transfer`, `apply`: 0.244, 0.033, 0.283, 0.041 s medians) are CPU submission intervals over asynchronous CUDA and must not be added causally. The whole install timer waits for GPU completion. The 8.179 GB baseline per unique PP×EP sender root is a real persistent cost; the short-lived peaks are separate.

## Conditional topology cost model

[`scripts/analyze_sparse_publication_topology.py`](../../scripts/analyze_sparse_publication_topology.py) reproduces the model from the final per-update summaries, one-H100 probe, and #698 relay. Sixteen unique PP×EP roots each own `E_root = 8.179 GB` of the 130.862 GB global routed-expert image; DP2 has replicas. For BF16 index/value encoding at change density `ρ`, payload is approximately `3ρE`: one four-byte index plus one two-byte value per changed element, compared with two dense bytes. Headers/padding are omitted; nonexpert weights remain dense in both arms.

```text
sparse − dense = fitted_nonwire_delta + (M_sparse − M_dense) × latency
               + (3ρ − 1) × E_root / effective_bandwidth
               + (1 − commit_overlap) × commit_seconds
```

`M` is the measured **expert broadcast collective** count per root: 832 dense and about 497 sparse at the observed density. The sparse count stayed effectively constant over the observed 1.4–2.4% range; extrapolating it is an assumption. The direct model solves the nonwire residual (0.325 s) from **one inter-node calibration point**: 5.3 ms median install advantage at 1.87% density, assumed 10 µs message latency, and `E_root / dense expert phase` = 23.59 GB/s as an **apparent service rate**. It is not a measured NIC bandwidth. The residual absorbs detection/encoding, receiver application, fixed collective effects and rate-model mismatch; it is not topology-independent. No direct-path density/topology was held out, so direct crossovers are sensitivity calculations without validated predictive accuracy. Useful baseline-commit overlap is unknown.

| Envelope (effective rate and assumed latency) | Install-only crossover | If commit cannot overlap | Reading |
| --- | ---: | ---: | --- |
| Same DC, 15 GB/s and 10 µs | ~13.6% density | ~12.3% | Sparse below, **if** the fitted residual transfers. |
| Measured-path apparent 23.59 GB/s and assumed 10 µs | ~2.4% | ~0.3% | Observed 1.87% lies between the two accounting choices; neither is a policy threshold. |
| Same DC, 30 GB/s and 30 µs | No positive crossover | No positive crossover | Dense predicted under this residual; unvalidated on that fabric. |
| One NVLink domain, 100–300 GB/s and 2–5 µs | No positive crossover | No positive crossover | Unvalidated extrapolation from inter-node NCCL; recalibrate locally. |

For cross-region, the retained 36-condition, one-sample-per-condition H100→S3 put/get→GB200 relay supplies object sizes, encode/staging/apply and two storage legs. Separate ordinary least-squares fits use 11 conditions per encoding and hold out 128 MiB at 23.1647% density. Held-out end-to-end predictions versus observations were dense **2.136 vs 1.811 s (+17.9%)**, indices **1.733 vs 1.479 s (+17.2%)**, and bitmap **1.025 vs 1.053 s (−2.7%)**. The 17–18% dense/index error and single-sample conditions preclude a precise crossover claim.

The new one-H100 BF16 probe used bit-exact application (including NaNs) and three synchronized repeats on representative Grug matrices and a 128 MiB bucket at 2.5%, 23%, and 50% change density. Sequential CPU D2H→detect→pack→H2D→apply→commit took **46, 103, 188 ms** on the 128 MiB bucket, versus **0.59 ms** for the GPU path at 2.5%; no NIC was used. Combining the fitted S3 put/get service with those CPU sparse stages gives **0.80 s at 2.5%** and **1.73 s at 23%** for one 128 MiB index object. A fitted **2.14 s dense reference** includes local stages from the *GPU* H100→GB200 relay. These are different candidate paths and hardware; their crossing is **not** a validated CPU sparse threshold. CPU bitmap construction was not measured. Full-model batching, concurrency, useful overlap and failure recovery remain unmeasured.

[NIXL](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md) abstracts memory and transport backends; it is not a physical link or an assumed speedup over NCCL. The repository and installed environment have no usable NIXL integration, and there is no equivalent-topology NIXL-versus-NCCL benchmark. The same-datacenter H100→GB200 private path was not available to validate. The next remote experiment must select and record the real backend/interface, staging, receiver apply and commit behavior before comparing strategies.

## Contained publisher design and next production step

Expose one run-level strategy: `dense`, `gpu_sparse_experts`, or an experimental `cpu_sparse_remote`. The trainer owns optimizer scheduling and the generation pause/drain/resume window. A publisher owns `prepare()` (schedule/receiver binding), `publish(version)` (transport and installed-byte acknowledgement), `commit(version)` (advance prior image only after receiver acknowledgement and resumed generation), `fail_closed(cause)` and `close()`, and returns common byte, density, collective, timing and memory metrics. `requires_initial_pause` can express the first-sync contract. The existing `ExpertBlockSync` already owns most of that lifecycle; add a thin factory/interface when a second concrete remote implementation exists. Keep baseline device, encoding, bucketing and transport private rather than adding independent public flags. Retain sparse routed experts plus dense nonexpert tensors for the GPU path; #698's all-tensor direct sparse probe was slower.

Keep dense as the production default. A run-level GPU sparse opt-in is warranted only when the approximately 94% expert-byte reduction is valuable enough to pay the 8.179 GB/root baseline and a roughly neutral or slightly longer sync; measure fabric contention or cost explicitly before claiming a throughput benefit. For cross-region, build a bounded remote publisher and compare dense versus CPU sparse under the actual backend and full-model async cadence before adoption.

## Provenance

- Work branch: `goal/sparse-publication-topology-20260920`, based on #701 `a130ffb990ff550be869df302ff60f0ed3ddfd7b`; seven #687 dependency commits preserved separately. Corrected performance runtime/config commit `d7598efccec73c62131fbe1c6164a9a82e294d11`; original qualification/pilot runtime `870f4cdfdec1b4f72cba9261a482926d81897060`. The analyzer and metric extractor were added after the corrected jobs, without runtime source changes.
- Corrected job identities: `/romain/sparse-topology-prod-preflight-20260920`, `/romain/sparse-topology-dense-prod-age4-20260920`, `/romain/sparse-topology-sparse-prod-age4-20260920`; all five tasks in each job terminal `succeeded`, failures/preemptions zero. Earlier qualification `/romain/sparse-topology-qualification-v3-20260920` and four-GPU gate `/romain/sparse-topology-small-gate-20260920` also succeeded. The pilot dense job reached `Training done!` and was manually marked complete only after a post-training pod deletion; no pilot timing is used as the decisive result.
- Durable evidence root: `s3://marin-us-east-02a/marin/users/romain/skyrl/sparse-publication-topology-20260920/evidence/`. The final jobs each have `{dense-prod-age4,sparse-prod-age4}-{trainer.err,resolved.json,summary.json}`; `production-comparison.json`, `topology-cost-model-v2.json`, `one-gpu-probe.json`, `small-distributed-gate.log` and `qualification-v3-*` are there too. Summaries preserve each individual update. The exact artifact hashes and launcher commands are in the #698 result comment.
- Cluster config supplied to the launcher: Marin reference revision `e49f36f2d7434776d9289a6bf3781f22b419ba39`, `lib/iris/config/cw-rno2a.yaml` blob `1d8d25a53d5f6124a1450cd3da4e9f0447eaf569`. It requests InfiniBand resources and host networking. Retained NCCL init logs did not show the chosen network plugin/interface, despite setting `NCCL_DEBUG=INFO`, so that detail is unverified.
