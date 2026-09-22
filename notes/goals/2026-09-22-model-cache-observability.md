# Measure cold Hugging Face model mirroring

## Objective

Make the existing MarinSkyRL model cache show where a cold mirror spends its time, so the next performance change can target measured cost. Keep the change small and useful on ordinary runs.

## Context

- [PR #715](https://github.com/marin-community/MarinSkyRL/pull/715) introduced the current cache path. In [`cloud/iris/hf_model_cache.py`](https://github.com/marin-community/MarinSkyRL/blob/3f9187b87ca0816a11b6c6db7a4f1be695c579db/cloud/iris/hf_model_cache.py), one lock holder downloads a pinned Hugging Face snapshot, builds and hashes its manifest, uploads files to S3, and writes the manifest last. Other tasks can hit the completed cache or wait for it.
- Romain asked for a Pareto-sized next step: always-on, low-volume phase timing; safe visibility into whether the task has an HF token and which S3 endpoint class it is configured to use; then use an ordinary cold run to decide whether further work is worthwhile. These are diagnostic goals, not evidence that any particular phase is slow.
- [Marin issue #9233](https://github.com/marin-community/marin/issues/9233) reports costly GPU waiting during a cold mirror in a different evaluation path. [Issue #9230](https://github.com/marin-community/marin/issues/9230) reports wasted work after an interrupted mirror in Levanter. Neither proves that MarinSkyRL has the same measured bottleneck.
- Start from the selected worktree containing this goal file. Read its root and applicable nested `AGENTS.md` and `TESTING.md`, and the shared development playbook. Preserve the dirty reference clone.

## Goal

- Add always-on, low-volume logs to the Hugging Face-to-S3 cache path. A cold publisher should expose elapsed time for download, manifest checking/hashing, and upload/publication, plus total time and enough context to distinguish a cache hit, a task waiting for the publisher, and the publisher itself. Use a monotonic clock. Phase boundaries should still help diagnose a job that ends before its final summary. Include model size or file count if already available without another expensive scan.
- On cold attempts, report only safe configuration indicators: whether `HF_TOKEN` is present and whether the configured S3 endpoint is the CoreWeave in-cluster endpoint, another endpoint, unset, or genuinely unknown. Distinguish an environment-derived indicator from the endpoint actually used by the storage client; do not label one as the other. Never log token values, credential values, or a full endpoint URL. Keep cache behavior and errors unchanged.
- Inspect the next suitable existing or normally scheduled cold run if one is available, and report the phase timings and any clear next action. If no such run exists, deliver the logging change and state that production timings remain unmeasured.

## Constraints & non-goals

- Favor a localized change to the cache path and directly relevant tests. Do not add a telemetry service, dashboard, per-file logging, or a new configuration flag.
- Do not implement a different transfer method, CPU prewarming orchestration, partial-mirror recovery, or checkpoint changes under this goal.
- Do not submit a new large download, GPU job, or other costly cloud workload for validation. Consult Romain before any such work, or if reliable endpoint classification requires substantial new plumbing. Explain the simpler options and recommend one; preserve progress and pause only the affected activity if no answer arrives.
- The code, any focused tests, and this goal file are repository deliverables. The developer's own session records follow their registered recording workflow and are outside that code-scope limit.

## Validation

- Start with cheap, targeted local checks of the cache path. Verify that a hit and a cold publication produce useful, low-volume diagnostics, and that a failed or interrupted attempt remains attributable to a phase. Check that sentinel credential values never appear in output. Add checked-in tests only for stable observable behavior; avoid assertions on incidental log wording or internal helper calls.
- Run the relevant CPU tests and required changed-file lint in the selected worktree. Before a draft PR, run the repository's lint-review pass once and address its findings. Reuse existing validation evidence when applicable.
- After ordinary code validation, request one final High-tier review with KIND `GOAL` under `~/repos/yonromai/env/playbooks/review-agents.md`. Give the reviewer this goal, Romain's scope decisions, the diff, and validation evidence. Follow the guide's bounded review policy; do not add review rounds without Romain's agreement.
- Open a draft PR when the agreed outcome and review are complete. For this supervised Marin task, hand back after current CI is terminal and current feedback is addressed. Do not start or re-arm the post-green `wait_for.py` monitor unless Romain asks for ownership through review or merge. Report any missing real-run timing as a limitation, not as a reason to launch an expensive run.
