# Compare exact delta weight-update designs used by RL systems

## Objective

Determine whether a current, externally demonstrated delta weight-update design can materially improve on MarinSkyRL's exact GPU sparse expert path. Audit relevant RL and inference systems, select at most two genuinely different candidates, and compare the promising candidates with dense publication and MarinSkyRL's current sparse implementation using controlled experiments.

## Context

Use a fresh isolated MarinSkyRL worktree and experiment branch. Recheck the live dependency stack before choosing the base; do not modify or rewrite existing PR branches. As of 2026-09-20, [#687](https://github.com/marin-community/MarinSkyRL/pull/687) is merged, [#688](https://github.com/marin-community/MarinSkyRL/pull/688), [#689](https://github.com/marin-community/MarinSkyRL/pull/689), and [#690](https://github.com/marin-community/MarinSkyRL/pull/690) remain open, and draft [#701](https://github.com/marin-community/MarinSkyRL/pull/701) is at `a130ffb990ff550be869df302ff60f0ed3ddfd7b`. Preserve dependency commits distinctly so experimental results can be attributed to the correct source.

Read the selected checkout's `AGENTS.md`, `TESTING.md`, and applicable nested guidance. Before Iris work, read the current Marin checkout's `lib/iris/OPS.md` and `.agents/skills/use-iris/SKILL.md`. Coordinate material shared accelerator use as required by the environment instructions.

The current evidence is retained in [issue #698](https://github.com/marin-community/MarinSkyRL/issues/698), especially the [fully asynchronous topology result](https://github.com/marin-community/MarinSkyRL/issues/698#issuecomment-5754383792). Its exact experiment branch is `goal/sparse-publication-topology-20260920` at `2c5aff75f00f6c9b2583559c956446bb0e0f2fda`, with the [full report](https://github.com/marin-community/MarinSkyRL/blob/2c5aff75f00f6c9b2583559c956446bb0e0f2fda/docs/experiments/sparse-publication-topology-2026-09-20.md) and reproducible analysis. Reuse its artifacts rather than rerunning controls whose source and configuration still match.

That work established:

- Exact GPU sparse index publication reduced routed-expert payload by 94.4% at roughly 1.9% changed density and preserved receiver bytes exactly.
- On the corrected same-datacenter, fully asynchronous Grug comparison, median expert installation improved by about 5 ms while full `sync_weights` became about 46 ms slower. One run per arm showed no attributable job-throughput gain.
- The current sender compares raw BF16 words against one GPU-resident acknowledged baseline, packs flat int32 positions plus BF16 replacements, routes changed expert data through Ahmad Qamar's expert-aware transport, and leaves shared and nonexpert weights dense.
- A GPU baseline costs about 8.2 GB per unique sender root. Baseline commit, detection, encoding, message count, receiver apply, GPU interference, and useful overlap can matter as much as payload size.
- CPU staging was measured only as bounded components. No production CPU-owned remote publisher, matched NVLink run, equivalent-topology NIXL comparison, or full-model cross-region run was qualified.

A prior survey inspected several alternative projects, but it was reconnaissance rather than a current, source-linked comparison followed by matched MarinSkyRL experiments. Relevant local reference clones include:

- `/home/romain/repos/verl-project/verl`
- `/home/romain/repos/NVIDIA-NeMo` and its NeMo RL source
- `/home/romain/repos/THUDM/slime`
- `/home/romain/repos/vllm-project/vime`
- `/home/romain/repos/radixark/miles`
- `/home/romain/repos/google/tunix`

Also inspect current official code, issues, and open PRs in other materially relevant systems, such as OpenRLHF, AReaL, SkyRL, Megatron-related RL integrations, vLLM, SGLang, NVIDIA Dynamo, and NIXL, when they contain an actual weight-publication mechanism. Popularity alone is not a reason to expand the survey. Prefer primary sources and exact revisions. Distinguish released or exercised behavior from proposals and comments.

The following decisions are settled:

- Updates must be exact. Lossy thresholds, quantization, approximate equality, and quality/performance trade-offs are outside this goal.
- The production workload of interest is fully asynchronous RL with nonzero staleness. A synchronous microbenchmark may isolate mechanisms but cannot decide the production result.
- Qualification and performance are separate. Use independent dense replay or another byte-level oracle for qualification, then remove that oracle from timed production-like runs.
- Expert-aware routing and sparse encoding are composable. Keep Ahmad's placement and direct expert routing as the transport baseline rather than treating them as competing alternatives.
- Dense remains the production default while the evidence is inconclusive. Do not add an adaptive selector or a universal density threshold.
- A negative result is useful. Do not tune workloads until a candidate wins or retain an implementation merely because another project uses it.

## Goal

First produce a concise comparison matrix of the best relevant external implementations. For each actual mechanism, record direct source links and pinned revisions, and explain:

- how it knows which weights changed: optimizer or routing metadata, tensor/block dirtiness, hashing, or elementwise comparison;
- the update unit and exact encoding: whole tensor, expert, block, bitmap, indices and values, or another representation;
- where the acknowledged baseline lives and how much memory it consumes;
- whether detection, packing, staging, and transport run on CPU or GPU and whether they overlap useful work;
- how weights are routed to the correct inference worker and transported;
- how the receiver applies an update;
- its version, acknowledgement, commit, and failure semantics;
- the topology and workload for which it was designed; and
- any public correctness or performance evidence.

Do not count renamed versions of dense broadcast, checkpoint reload, or the existing MarinSkyRL algorithm as new candidates. Explain when a widely used library does not implement true deltas; that negative finding prevents cargo-culting its surrounding architecture.

From this audit, select at most two candidates with a concrete reason they could improve a measured MarinSkyRL bottleneck. Likely mechanisms worth checking include deriving dirty experts or blocks from optimizer/routing state to avoid a full comparison, exact bitmap or block encoding to reduce index and collective overhead, and pipelining detection, packing, staging, and transfer. These are suggestions, not required selections. Prefer the externally supported mechanisms that best fit the evidence.

Prototype only enough of each selected design to measure its real costs. Keep experimental implementations behind the existing publication boundary. Reuse one common measurement contract and the existing exactness oracle. Avoid spreading candidate-specific branches through the trainer or building a permanent abstraction before a second implementation proves that it is useful.

Compare surviving candidates against:

1. Ahmad's dense expert-aware publication;
2. #701's GPU sparse-index expert publication; and
3. each other, where both address the same topology.

Use representative Grug expert shapes and densities that cover the observed 1.4–2.4% range and lower densities expected later in training. Add higher densities only as needed to identify crossover behavior. Measure raw and encoded bytes, changed density, baseline memory, temporary memory, detection, packing, staging, collective or message count, transfer, receiver apply, acknowledged commit, full publication interval, GPU and CPU utilization where available, and useful overlap. In production-like runs also measure accepted useful tokens per wall time, trainer and rollout waiting, realized staleness, retries or discarded groups, and whether publication is hidden by asynchronous work.

For every candidate that survives small-scale measurements, run a byte-exact qualification and then a matched production-like comparison with verification disabled. Freeze the model, source, seed, data, topology, update schedule, serving limits, optimizer placement, and async cadence. Compare resolved configurations mechanically before launch. Use enough post-warmup publications to expose normal variation and retain individual samples. Do not attribute differences in accepted async work to the publication mechanism without the appropriate normalization and caveat.

If a CPU-owned or remote design is selected, measure the CPU comparison, packing, D2H/H2D staging, transport, application, and overlap on the actual available backend where practical. Reuse #698's cross-region artifacts. Do not perform another material cross-region transfer without first telling Romain the expected bytes, cost, and uncertainty it resolves. Do not claim NIXL, object storage, TCP, NCCL, NVLink, InfiniBand, or RoCE performance for a topology that was not actually exercised.

Update #698 with a decision-oriented result. The visible section should state what competing systems actually do, which candidates were selected and why, matched results, implementation and topology limits, and the recommended next production step. Put exact revisions, commands, job IDs, raw artifact locations and hashes, larger tables, and negative results in collapsible provenance or a linked report. Preserve enough evidence for another engineer to reproduce the conclusion.

## Constraints and non-goals

- At most two new designs may reach a full multi-node Grug comparison. Use existing controls when they match exactly; otherwise run the minimum matched control needed for a valid comparison.
- Before adding another full-topology candidate, repeated full-topology replicas, or a material cross-region transfer beyond that scope, explain what uncertainty it resolves and consult Romain. Continue independent work while awaiting an answer; preserve progress and pause only the affected activity.
- Do not modify, rebase, or merge #688–#690 or #701 as part of this experiment. Stack experimental work cleanly on the live dependency heads and report when dependency movement affects the result.
- Do not turn experimental candidates into production code, change the production default, or open several implementation PRs. If a design clearly wins, specify the smallest coherent follow-up PR rather than polishing every prototype for merge.
- Do not implement lossy deltas, adaptive runtime selection, automatic recovery, checkpointing, or a general transport framework.
- Do not require every CPU/GPU, expert/all-weight, encoding, and transport combination. Test only combinations justified by the external audit and a measured MarinSkyRL bottleneck.
- Keep retained tests Pareto-principled. Prefer one public exact round trip or independent numerical oracle over tests of private helpers, registration, strings, or every experiment parameter.
- There is no fixed total time or retry count. Diagnose ordinary launch, dependency, queue, and runtime failures while useful progress remains. Before slow or costly work, exercise the cheapest representative workflow in its real launch environment. Preserve costly outputs and lengthen polling intervals as expected duration becomes clear.
- Do not change shared networking, credentials, or Iris controllers. Do not make the unavailable same-datacenter H100-to-GB200 path a blocker.

## Validation

1. Reproduce the relevant #698 numbers from retained artifacts. Confirm the production recipe and identify the earlier capped/offloaded pilot so it cannot be mistaken for the corrected baseline.
2. Complete the source-linked external comparison and shortlist before writing a substantial prototype. Verify important claims against current repository code and open work, not summaries or project names.
3. Use CPU checks and small deterministic tensors to validate codec arithmetic and exact application. On one development H100, compare selected mechanisms with representative Grug shapes across the relevant density range, including raw BF16 edge cases. Reject clearly dominated candidates before distributed work.
4. Exercise each surviving candidate on the smallest distributed trainer/live-receiver topology that covers its communication behavior. Confirm exact destination bytes over at least two successive updates and verify that the acknowledged first update becomes the next baseline.
5. Before a full run, execute the exact launch method as a short preflight and inspect the resolved configuration. Confirm serving concurrency, optimizer placement, source revision, model revision, staleness, buffer size, and verification setting. Use `cw-rno2a` for H100 work if current configuration and capacity still make it suitable; use interactive priority for development and production priority for the final bounded comparisons.
6. Run byte-exact qualification separately from matched fully asynchronous performance runs. Preserve raw samples and terminal Iris status. Ensure profiling does not materially alter the unprofiled fast path.
7. Run targeted repository tests and changed-file formatting for code retained on the experiment branch. Experimental analyses and sweeps do not each need permanent pytest coverage.
8. Before publishing the final #698 update, request one High-tier `KIND=GOAL` review following `/home/romain/repos/yonromai/env/playbooks/review-agents.md`. Give the reviewer this goal, the external comparison, candidate rationale, final diff, resolved configurations, and raw evidence. Ask it to focus on missed prior art, experimental confounders, exactness, topology mismatch, misleading throughput attribution, and unnecessary implementation or tests. Apply material findings with judgment; do not start another review round unless a consequential correction cannot be validated directly.

Finish when all submitted jobs are terminal, review findings are addressed, and #698 contains a source-linked comparison, reproducible experimental evidence, and a clear recommendation about whether MarinSkyRL should replace, supplement, or retain its current exact sparse-index design. Do not start Marin's post-green `wait_for.py` monitor. Report the experiment branch and revisions, candidate designs, commands and job identities, artifact locations, measured results, review feedback applied, and remaining unvalidated assumptions.
