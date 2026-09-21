# Sparse weight publication across topologies (2026-09-20)

This experiment isolates the publication choice for fully asynchronous Grug RL. It does not change the production default. The real H100 comparison supports GPU-resident sparse expert publication as a **bandwidth and byte-saving option** on the measured same-datacenter fabric at low change density, but does not show a job throughput gain. Dense publication remains the safe run-level baseline. CPU-resident sparse remote publication is a conditional design candidate, not a validated production transport.

| Topology | Present decision | Evidence and condition |
| --- | --- | --- |
| One NVLink domain | Use dense until a matched trainer measurement shows otherwise. | The older single-H100 direct-copy microbenchmark in [#698](https://github.com/marin-community/MarinSkyRL/issues/698) found dense faster than sparse encodings. The full trainer experiment crosses nodes, so the model's NVLink prediction is extrapolation only. |
| H100 nodes in one datacenter | Offer a run-level GPU sparse-expert choice when effective transfer is bandwidth-limited and expert changes remain small; do not promise throughput improvement. | At 1.7% median changed density, 20 post-warmup real updates reduced expert bytes 130.862 to 6.811 GB and expert install 0.441 to 0.430 s, but whole-step wall sums were 1190.78 vs 1188.70 s. |
| Cross-region trainer/inference | Keep dense as the proven fallback; prototype a CPU sparse sender/receiver behind one publisher boundary before adoption. | A prior one-object H100→S3→GB200 relay favored bitmap and indices at 23.16% density, and a new one-H100 CPU staging probe bounds local costs. There is no matched full-model remote async run or measured direct NIXL path. |

## Experiment and correctness

Both performance arms used source `870f4cdfdec1b4f72cba9261a482926d81897060`, model revision `6808fe5c219471517bd51df35addefd38ebebf89`, seed 17, the same GSM8K data/configuration, 25 updates, maximum staleness 4, buffered groups 32, 160 generation workers, four responses per prompt, and an 8192/4096 context/output limit. Topology was four 8×H100 trainer nodes (PP2, EP8, DP2) and one 8×H100 vLLM node (DP8, EP8) on `cw-rno2a`. This follows the fully async operating point from [Marin #8955](https://github.com/marin-community/marin/issues/8955); source-specific mask/parser settings differ. The only publication override was `generator.expert_block_sync.encoding=sparse_index` in the sparse arm. Both disabled independent replay. Steps 1–5 are excluded as startup; every table value below is from steps 6–25. Async scheduling changed the actual rollout mix despite the same seed.

Correctness was established separately on the same five-node Grug topology by `/romain/sparse-topology-qualification-v3-20260920`: initial dense seed and four sparse publications, `max_staleness_steps=0`, `verify=true`. Independent dense replay compared every receiver's raw BF16 bytes against a dense reference, including data-parallel replicas, and matched the expected byte schedule for all five versions; the job succeeded naturally. The four-H100 gate also checked deliberate corruption detection. Qualification timing includes replay and is not used as performance timing.

| Post-warmup metric (median; IQR when useful) | Dense | GPU sparse |
| --- | ---: | ---: |
| Full `sync_weights` interval | 3.987 s (3.733–4.166) | 4.199 s (3.903–4.268) |
| Expert-block install | 0.441 s (0.440–0.443) | 0.430 s (0.427–0.434) |
| Expert transfer phase | 0.347 s | 0.344 s |
| Sender baseline commit (after acknowledged resume) | — | 0.0199 s driver; 0.0120 s rank max |
| Global logical expert bytes / encoded bytes | 130.862 / 130.862 GB | 130.862 / 6.811 GB (5.621–7.719) |
| Changed expert density | — | 1.7% (1.4–2.0%) |
| Global expert collectives | 13,312 | 7,956 |
| PyTorch sender peak GPU allocation | 10.844 GB | 19.047 GB (includes ~8.179 GB resident prior-image baseline per root) |
| Sender short-lived GPU peak over pre-publication allocation | — | 29.18 MB median |
| Receiver short-lived GPU peak | — | 12.63 MB median |
| Sender peak process RSS (`ru_maxrss`, cumulative) | 38.727 GB | 38.768 GB |
| Step duration | 55.891 s (52.668–64.012) | 58.282 s (53.152–65.127) |
| Sum of step durations, 20 updates | 1190.78 s | 1188.70 s (−0.17%) |
| Generation-buffer wait | 39.408 s | 41.384 s |
| Training interval | 12.684 s | 12.036 s |
| Mean rollout staleness | 3.172 | 2.828 (maximum 4 in both arms) |
| Stale attempts rejected and retried, sum | 50 | 69 |
| Accepted response tokens / summed step time | 4,193/s | 3,986/s |

The accepted-token rate above is `sum(generate/avg_num_tokens × 128) / sum(timing/step)`. It excludes prompt and padding tokens, and the arms accepted different numbers of response tokens (4.993M vs 4.738M); it is **not** an exact total-training-token rate or an accuracy result. The trainer retries stale groups rather than counting them as accepted. The pause-call timer measures only the call to pause (median 0.441 vs 0.561 s), not the whole pause-to-resume window. Full `sync_weights` includes optimizer offload (mean 3.061 vs 3.128 s), pause/drain, transfer, resume and sparse commit. Phase CPU submission timers overlap asynchronous CUDA work, so they must not be summed as a causal breakdown. The approximately 0.15 s mean full-sync difference is similar to offload and pause-call variation; the measured expert install advantage is only 11 ms. The shorter wire payload was mostly hidden by other work and async variation. It did not improve observed job throughput.

The dense job logged all 25 updates and `Training done!`; a pod was deleted during post-training shutdown, and `iris job complete` closed the leftover cleanup state. Iris reports `succeeded`, one preemption, zero failures. Sparse and qualification jobs succeeded without intervention.

## Calibrated, conditional cost model

The reproducible calculator is [`scripts/analyze_sparse_publication_topology.py`](../../scripts/analyze_sparse_publication_topology.py). A direct-path root owns 8.179 GB of the 130.862 GB global routed-expert image (16 unique PP×EP roots). For a BF16 index/value encoding at density `ρ`, the payload approximation is `3ρE`: one four-byte index plus one two-byte BF16 value for each changed element, compared with two dense bytes per element, with padding and headers excluded. The direct install difference is modeled as

```text
sparse − dense = fitted_nonwire_delta + (M_sparse − M_dense) × latency
               + (3ρ − 1) × E_root / effective_bandwidth
               + (1 − commit_overlap) × commit_seconds
```

`M` is the number of 128 MiB-bucket collectives per root; it is 832 dense and about 497 sparse at the observed density. The sparse publication still sends nonexpert weights densely. The fit uses the measured 11 ms install advantage at median density 1.7%, a *chosen* 10 µs message latency, and `E_root / dense_expert_phase` = 23.56 GB/s as an **apparent service rate**, not a NIC throughput measurement. The residual `fitted_nonwire_delta` absorbs detection/encoding, receive application, fixed collective work, dense nonexpert work and any mismatch between the rate assumption and reality. It cannot be carried to a different topology as an invariant CPU cost. Sparse commit is 19.9 ms median at the driver; overlap with resumed generation is unknown.

| Envelope, assumed effective GB/s and message latency | Install-only crossover density | With no useful commit overlap | Interpretation |
| --- | ---: | ---: | --- |
| Same DC, 15 GB/s and 10 µs | ~13.9% | ~12.7% | Sparse below, under the fitted residual assumption. |
| Measured-path apparent 23.56 GB/s and assumed 10 µs | ~2.8% | ~0.9% | Observed 1.7% lies between these accounting choices; do not treat either as an empirical policy threshold. |
| Same DC, 30 GB/s and 30 µs | No positive crossover | No positive crossover | Dense predicted for the whole physical density range under this residual. |
| One NVLink domain, 100–300 GB/s and 2–5 µs | No positive crossover | No positive crossover | Unvalidated extrapolation: the residual was fitted on inter-node NCCL and could change. |

The measured collective count stayed constant over the observed 1.16–2.34% density range; holding it fixed beyond that range is another extrapolation. The crossover moves by more than an order of magnitude across plausible effective rates. Message latency, bandwidth and overlap in this table are assumptions except for the apparent measured-path rate. This model explains sensitivity; a new topology needs its own measured calibration and repeat runs. It does not justify an automatic selector.

For the remote path, the older 36-point, one-sample-per-condition relay (H100 sender, S3 put/get, GB200 receiver) supplies measured object bytes, local encode/staging/apply, and two storage legs. For each encoding, an ordinary least-squares fit uses 11 conditions and holds out the 128 MiB, 23.1647% condition. Dense features are intercept and object MiB; sparse features add logical MiB. These coefficients describe storage-service time, not physical link bandwidth. Held-out end-to-end predictions versus observation were dense **2.136 vs 1.811 s (+17.9%)**, indices **1.733 vs 1.479 s (+17.2%)**, and bitmap **1.025 vs 1.053 s (−2.7%)**. This validation error and the single-sample relay rule out a precise remote crossover claim.

The new one-H100 BF16 probe measured bit-exact GPU sparse application and CPU D2H→detect→pack→H2D→apply→commit for representative matrices and a 128 MiB bucket at 2.5%, 23%, and 50% density (three synchronized repeats). At 128 MiB, CPU stages took 46, 103, and 188 ms versus 0.59 ms for the GPU sparse path at 2.5%; no NIC was used. Composing the fitted S3 put/get service cost with these CPU phases predicts **0.80 s at 2.5%** and **1.73 s at 23%** for one sequential 128 MiB index object, versus a fitted **2.14 s dense** object. The resulting ~32% index crossover is a sensitivity calculation only: the CPU probe used an H100, the relay receiver was GB200, the S3 fit misses the held-out point by 17–18% for dense/indices, and full-model batching, concurrency, overlap and failure recovery were unmeasured. Bitmap was faster than indices in the held-out GPU relay, but CPU bitmap construction was not measured. Do not promote either remote encoding from this result.

NIXL is a transfer API over different memory and transport backends, not a replacement physical network or an assumed acceleration over NCCL. No equivalent-topology NIXL-versus-NCCL measurement exists here; the repository/installed environment has no usable NIXL integration. A same-datacenter H100→GB200 private path was not available to validate. For a direct remote path, measure the selected NIXL backend or TCP route, CPU pinned-memory staging, receiver application and commit under the actual topology before comparing it with NCCL or S3.

## Contained publisher design

Keep one run-level strategy selection: `dense`, `gpu_sparse_experts`, or an experimental `cpu_sparse_remote`. The trainer owns the optimizer offload and generation pause/drain/resume window. A publisher owns `prepare()` (schedule and receiver binding), `publish(version)` (transport and installed-byte acknowledgement), `commit(version)` (advance prior image only after all receivers acknowledged and generation resumed), `fail_closed(cause)` and `close()`. It returns one common metrics record for bytes, density, collectives, phase time and memory. A capability such as `requires_initial_pause` lets the trainer keep the first-sync contract without knowing encoding. The current `ExpertBlockSync` already has most of this lifecycle and exact failure behavior; a thin interface/factory can wrap it and the dense path when a second concrete remote implementation exists. Avoid exposing baseline device, sparse representation, bucketing and transport as independent public flags. Retain sparse routed experts plus dense nonexpert tensors for the GPU path; #698's all-tensor direct sparse microbenchmark was slower.

The next production step is a small run-level opt-in for GPU sparse routed experts, with resident-memory budgeting and the existing fail-closed semantics. Keep dense as default. For cross-region deployment, build a bounded remote publisher prototype and compare it against dense on its actual backend and full-model async cadence before a production switch.

## Provenance

- Work branch: `goal/sparse-publication-topology-20260920`, based on #701 `a130ffb990ff550be869df302ff60f0ed3ddfd7b` with the seven #687 dependency commits preserved separately. Runtime commit for all full-topology arms: `870f4cdfdec1b4f72cba9261a482926d81897060`.
- One-GPU probe: `/romain/sparse-topology-one-gpu-v2-20260920`; 4-GPU gate: `/romain/sparse-topology-small-gate-20260920`.
- Qualification: `/romain/sparse-topology-qualification-v3-20260920`; performance: `/romain/sparse-topology-dense-age4-20260920` and `/romain/sparse-topology-sparse-age4-20260920`. The two failed qualification setup attempts were canceled before v3; they supplied no result.
- Frozen recipe: `cloud/iris/configs/sparse_publication_topology.yaml`, `trainer.logger=console`, `+trainer.fully_async.max_buffered_groups=32`. Qualification adds `trainer.max_steps=4`, `trainer.fully_async.max_staleness_steps=0`, `generator.expert_block_sync.verify=true`, and `encoding=sparse_index`; sparse performance adds only `encoding=sparse_index`. Each run had unique output paths and job names.
- Durable raw evidence root: `s3://marin-us-east-02a/marin/users/romain/skyrl/sparse-publication-topology-20260920/evidence/`. Full-job files are `{qualification-v3,dense-age4,sparse-age4}-{trainer.err,resolved.json,summary.json}`; the one-GPU file is `one-gpu-probe.json`; the small gate is `small-distributed-gate.log`. Each summary preserves all per-update metrics. Raw #698 relay inputs were retained separately in the source revision cited there.
