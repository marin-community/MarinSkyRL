---
name: rl-standard-job-cleanup
description: Preserve, validate, and hand off a terminal standard MarinSkyRL run backed by dataset rows and programmatic rewards, without an agent harness or per-trial artifacts. Use after completion or failure, or before an authorized cancellation. Use rl-agentic-job-cleanup for agent trajectories.
---

# Clean up a standard RL run

This workflow preserves evidence. It does not authorize cancellation, publication, registration,
deletion, or database mutation.

## Workflow

1. **Classify the run.** Read the resolved configuration, not the name. Confirm that the run has no
   agent harness, sandbox, or per-trial artifacts.
2. **Establish terminal state.** Pair controller state with durable outputs and the run's configured
   step budget. Preserve the first causal error for failed jobs.
3. **Capture the full chain.** Collect logs from every restart or resume plus training metrics,
   checkpoints, export metadata, and the resolved launch command/configuration. Do not treat absent
   trials or verifier artifacts as missing evidence for a standard run.
4. **Validate candidates.** Check shard completeness, step consistency, configuration/tokenizer
   metadata, metric coverage, and the last saved step. Derive model size from exported tensors.
5. **Select only by declared policy.** If the experiment defines no selection criterion, report all
   valid candidates and leave selection unresolved. State the metric window and chain coverage for
   any selected artifact.
6. **Hand off or publish only with explicit authority.** Verify destination, visibility, provenance,
   and a clean secret scan. Confirm whether registration applies before creating a row.
7. **Reclaim only with separate authority** and only after the destination has been verified.

Before an authorized cancellation, capture live process state, per-rank device utilization, and
in-flight outputs that will disappear. Do not terminate an unexplained reproducing failure before
recording the evidence needed to diagnose it.

## Completion record

Report terminal state, source revision, resolved configuration, preserved locations, validation
results, selected artifact when authorized, first failure cause, resume safety, and remaining work.
