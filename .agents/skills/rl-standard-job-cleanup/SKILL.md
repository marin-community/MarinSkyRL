---
name: rl-standard-job-cleanup
description: Preserve, validate, and hand off a completed or failed STANDARD (non-agentic) MarinSkyRL training job — a parquet-only RL run with no agent harness, no sandbox environment, and no per-trial artifacts. Use after such a job reaches a terminal state or before an authorized cancellation. For a run that executes agent trajectories, use rl-agentic-job-cleanup instead.
---

# Standard RL Job Cleanup

Use this workflow to preserve the evidence needed to understand, reproduce, publish, or resume a
**standard (non-agentic)** RL run — one whose training data is a parquet task source and whose
rollouts are model generations scored by a programmatic reward, with no agent harness, no sandbox
environment, and no per-trial artifacts. It does not authorize cancellation, deletion, database
mutation, or model publication by itself.

**Choosing between the two cleanup skills.** Read the run's resolved configuration, not its name: if
it defines an agent harness / sandbox environment and writes per-trial outputs, use
`rl-agentic-job-cleanup`. If it does not, use this one. Applying the agentic skill here sends you
looking for trial directories, verifier outcomes, and a companion trace dataset that were never
produced, and their absence is not evidence of a fault.

## 1. Confirm the lifecycle state

- Query the current control plane and the job's durable outputs. Do not infer completion from a
  single pod, a stale log, or the absence of a terminal log line — a clean exit reaps its workers
  and may emit nothing.
- Confirm completion against the run's own step budget, not against elapsed time or an earlier
  estimate. A run that finished early or late is a different finding from one that finished.
- Identify the exact source revision, resolved launch configuration, model, data, resource layout,
  and artifact destinations that produced the job.
- Classify the run as completed, failed, cancelled, or indeterminate. For a failure, retain the
  first causal error as well as the terminal controller state.

## 2. Preserve durable evidence

A standard run's evidence surface is narrower than an agentic one's, and the difference is
load-bearing:

- training metrics and the trainer log;
- checkpoints and export metadata;
- the resolved launch command and configuration;
- worker logs relevant to any failure.

There are **no** trials, rewards-per-trial, verifier outcomes, sandbox records, or trace/literal
evidence. Do not report their absence as missing data.

**Cover the whole log chain.** A run that restarted or resumed writes one log per link, and the
early links often carry no training records at all. A metric series computed from a single
mid-chain log silently under-covers the run and can support a wrong conclusion. Establish how many
links exist before reading any series from them, and prefer a durable artifact listing over a log
stream when measuring throughput — log streams may be rate-limited and undercount high-frequency
events during a burst.

Copy only what diagnosis or archival requires, retain its provenance, and verify transfers before
treating the source as disposable.

## 3. Validate a candidate result

Before calling a checkpoint or export usable, verify it is complete for the requested next action:

- checkpoint shards and metadata agree on the same completed step;
- tokenizer and configuration files match the intended model export;
- the metric series covers the claimed interval without unexplained terminal errors;
- the exported step is at or below the last step the run actually saved.

Select an artifact only by an experiment-defined criterion. If none is supplied, report the complete
candidates and leave selection to the user rather than silently choosing a "best" one. When a
criterion is supplied, state the window it was computed over and whether the chain was fully covered
— a selection rule is only as trustworthy as its input series.

**Derive any size or shape descriptor from the exported weights themselves**, never from the run
name, the base-model name, or a config field that was inherited rather than measured. Names carry
stale or aspirational values; the weights are the artifact being published. Cross-check a parameter
count computed from tensor shapes against an independent estimate, and when they disagree
materially, prefer the measured value and say so.

## 4. Hand off or publish only with authority

- Publish or register an artifact only when the requested scope authorizes it. Verify the intended
  destination, visibility, provenance, and secret-free metadata first — scan the staged directory
  for credentials before upload, not after.
- Confirm whether the run's series is registerable at all before registering it. Some series are
  publish-only by policy, and an unwanted registry row is harder to retract than to avoid.
- Never delete checkpoints, remote objects, or database records as a side effect of cleanup.
  Reclaiming storage is a separate, explicitly authorized action, taken only after the published
  artifact is confirmed at its destination.
- For a failed run, distinguish infrastructure, configuration, and model/training failures. State
  whether a resume is safe and the smallest corrective action required.

## Before any authorized cancellation

If cleanup precedes a cancellation, capture the evidence that dies with the job before it stops:
live process state on the suspect ranks, per-rank device utilization, and in-flight outputs. And do
not cancel a run that is exhibiting a signal you cannot yet explain — a live job is a reproducing
instance, and an unrelated but valid reason to stop it still destroys the reproduction. Record what
diagnostic capability a cancellation costs, so the tradeoff is visible to whoever authorized it.

## Completion record

Return the terminal state, source revision, resolved configuration, preserved artifact locations,
validation result, selected artifact if authorized, and any remaining action. State plainly which
evidence classes do not exist for a standard run rather than leaving them blank. Keep the record
factual and tied only to the observed run and its declared experiment policy.
