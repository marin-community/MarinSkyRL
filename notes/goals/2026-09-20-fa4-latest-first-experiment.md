# Latest-first FlashAttention 4 experiment

## Objective

Determine whether MarinSkyRL should adopt a newer Megatron/attention dependency cohort with FlashAttention 4 for Grug training. Start from the newest plausible versions, measure performance and numerical behavior on the paths Marin uses, and publish a clear recommendation without creating another production PR in this pass.

## Context

- Work from this isolated worktree, based on MarinSkyRL `main` after [#705](https://github.com/marin-community/MarinSkyRL/pull/705). Read the repository instructions, testing policy, and the applicable Iris runbooks before changing files or launching jobs.
- The current x86_64 Megatron closure uses Torch 2.13, CUDA 13.2, Transformer Engine 2.11, Megatron Core 0.18, Megatron Bridge 0.6, and FlashAttention 2.8.3. #705 restored context-parallel training after Transformer Engine 2.11 rejected FlashAttention 2.8.4.
- [Upstream SkyRL #2132](https://github.com/NovaSky-AI/SkyRL/pull/2132) provides the main reference implementation. It keeps FA2 as the default and makes FA4 opt-in through combined FA2/FA4 wheels. It found meaningful B300 kernel gains, no measurable benefit in a short-sequence B200 RL run, and did not promote Hopper because FA4 backward stalled there.
- As of 2026-09-20, the newest relevant releases are [Transformer Engine 2.19](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.19), [Megatron Core 0.19.2](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.19.2), [Megatron Bridge 0.6.2](https://github.com/NVIDIA-NeMo/Megatron-Bridge/releases/tag/v0.6.2), and FlashAttention 4 beta31. Recheck this at execution time and record the exact versions tested. Core 0.19.2 and Bridge 0.6.2 track Transformer Engine 2.18, while Transformer Engine 2.19 adds FA4 support for supported context-parallel communication modes. Treat TE 2.19 on that released Megatron cohort as a deliberate compatibility experiment, not a vendor-qualified combination.
- Preserve these supervising-session decisions:
  - Try the newest plausible package cohort first. Back off selectively only when evidence identifies an incompatibility; do not default to the smallest version bump.
  - Avoid repeated independent refreshes by evaluating the tightly coupled Megatron and attention packages together.
  - Performance and silent numerical degradation matter most. Loud failures are useful evidence and may be repaired; a run merely avoiding an exception is insufficient.
  - Keep the already-qualified Torch 2.13, CUDA 13.2, and Marin vLLM versions fixed unless a demonstrated hard incompatibility makes that impossible.
- The sparse weight-transfer work and its stacked prerequisite PRs are separate. Do not edit, rebase, or depend on their worktrees or PR branches.

## Goal

Create a reproducible experimental branch that answers whether a latest-first FA4 stack is viable and worthwhile for Marin's Grug workloads.

1. Inspect the current Marin closure, upstream SkyRL #2132, and the relevant NVIDIA and FlashAttention release metadata. Choose the newest plausible coupled cohort rather than upgrading packages independently. The initial candidate should use the latest released Megatron Core and Bridge, Transformer Engine 2.19 for CP-aware FA4, and the latest compatible FA4 beta. Include ModelOpt, CUTLASS DSL, Quack, and other directly coupled packages when their dependency contracts require it.
2. Resolve and install that cohort while holding Torch 2.13, CUDA 13.2, and the current Marin vLLM wheel fixed. Reuse upstream packaging machinery and supported package interfaces where practical. Do not introduce runtime monkey patches, broad compatibility shims, or a second dependency system merely to force resolution.
3. Diagnose failures and repair reasonable incompatibilities. If the newest cohort does not work, isolate the edge before stepping back. Useful fallback probes include the vendor-aligned TE 2.18 cohort and upstream SkyRL's known FA4 beta28 combination. Preserve the newest viable combination and explain every retreat from a newer version.
4. Compare three arms with otherwise matched configuration and inputs:
   - the current Marin stack with FA2;
   - the new dependency cohort with FA2 explicitly selected;
   - the same new cohort with FA4 explicitly selected.

   This must separate the effect of the general dependency refresh from the effect of FA4 itself. Capture explicit backend-selection evidence; a silent fallback to FA2 does not count as an FA4 result.
5. Measure the behavior Marin cares about. At minimum, exercise Grug Megatron forward, backward, and an optimizer update with context parallelism, packed inputs where the real path uses them, and representative H100 and GB200 shapes. Record pointwise output and gradient differences against the matched FA2 arm, finite-state checks, memory, kernel or attention time, training-step time, and enough end-to-end timing to show whether an attention improvement matters to the RL workload.
6. Keep the implementation experimental and easy to remove. It may be rough where that speeds learning, but do not leave unexplained hacks, defensive fallback machinery, or tests that assert package internals. Prefer disposable probes for import and backend-selection checks and a few behavior-heavy tests for durable coverage.
7. Update [MarinSkyRL issue #4](https://github.com/marin-community/MarinSkyRL/issues/4) so it reflects the current CUDA 13.2 stack rather than its obsolete pre-#561 assumptions. Include the exact dependency matrix, links to reproducible commits and jobs, backend-selection proof, numerical and performance results, known hardware or topology limits, and one recommendation:
   - proceed with a production migration;
   - keep FA2 and defer FA4; or
   - pursue one specifically identified missing prerequisite.

   If migration is recommended, propose a small reviewable landing sequence, but do not open the production PRs in this pass. Preserve the prototype branch and any candidate wheel provenance needed to reproduce the result.

## Constraints and non-goals

- Keep Torch 2.13, CUDA 13.2, and the current Marin vLLM source/wheel fixed. If one must change, stop that dependent experiment, preserve the evidence, and consult Romain with the incompatibility and the smallest credible options.
- Do not change sparse weight synchronization, the stacked weight-transfer work, or vLLM behavior.
- Do not attempt an all-repository FA4 migration. This goal covers MarinSkyRL's active Grug Megatron path and the dependency cohort required for it.
- FA2 remains the operational default during the experiment. Do not silently change production behavior.
- Do not hide unsupported hardware or topology behind automatic fallback. Detect and report which backend actually ran.
- There is no one-hour deadline and no fixed retry count. Continue useful diagnosis and repair when a failure has a credible next step. Avoid repeating an unchanged expensive run.
- Short H100 and GB200 experiments and the native builds required to test the selected cohort are authorized. Start with cheap checks and small accelerator probes. Proceed to one representative Grug run per useful arm only after the earlier stage passes. Reuse artifacts and build caches when sound.
- Do not transfer large artifacts across regions. Use existing cluster-local model and dependency sources where possible. Do not change or restart shared Iris controllers or clusters.
- Run H100 probes in `cw-rno2a` and GB200 probes in `cw-us-east-08a`, using development or interactive priority. Use the priority already configured by the closest representative Grug workflow for its final run; do not raise priority only to shorten the queue.
- Negative evidence can satisfy the goal. Do not manufacture a production change when FA4 is slower, numerically suspect, unsupported, or operationally too costly.

## Validation

Validate progressively and preserve the exact commit, lock, package versions, command, hardware, topology, and terminal result for every accelerator claim.

1. Before a native build or accelerator job, prove that the lock resolves, inspect wheel availability and metadata, and exercise the post-build installation/import path against an existing artifact or minimal fixture where possible.
2. Run CPU packaging and lock checks. Keep only tests that enforce a stable dependency or behavior contract; import probes and package-version exploration belong in disposable scripts or recorded commands.
3. Run a one-GPU H100 and one-GPU GB200 preflight for installation, imports, kernel initialization, and explicit FA2/FA4 backend selection. Use bounded execution and preserve the first causal failure.
4. Run small matched forward/backward/optimizer comparisons on each supported architecture. Use an independent or established FA2 reference and report max and mean deviations; do not weaken repository tolerances without Romain's agreement.
5. Exercise context parallelism on the smallest topology that proves the CP path and confirms FA4 was selected. Then run the shortest representative Grug workload needed to measure training behavior. Scale farther only when the smaller result leaves a material production question unanswered.
6. Compare the three experimental arms with warmups and repeated steady-state samples sufficient to distinguish a real change from noise. Report confidence or observed variance; do not present a single timing as a throughput conclusion.
7. Run the repository's changed-file lint and relevant CPU tests for any retained prototype code. Do not run the entire test suite merely because it exists.
8. Before publishing the issue update, request one final High-tier review with `KIND=GOAL` following `~/repos/yonromai/env/playbooks/review-agents.md`. Give the reviewer this goal, the exact diff or prototype, commands, accelerator evidence, and proposed conclusion. Address material findings without starting extra review rounds.
9. Hand back after issue #4 is updated, the prototype and evidence are preserved, and the recommendation is supported. Do not start Marin's post-green `wait_for.py` monitor.
