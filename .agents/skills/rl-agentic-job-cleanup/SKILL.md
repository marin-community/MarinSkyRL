---
name: rl-agentic-job-cleanup
description: Preserve, validate, and hand off a completed or failed agentic MarinSkyRL Iris training job. Use after an Iris RL job reaches a terminal state or before an authorized cancellation.
---

# Agentic RL Job Cleanup on Iris

Use this workflow to preserve the evidence needed to understand, reproduce, publish, or resume an agentic RL run. It does not authorize cancellation, deletion, database mutation, or model publication by itself.

## 1. Confirm the lifecycle state

- Query the current Iris control plane and the job's durable outputs. Do not infer completion from a single pod or a stale log.
- Identify the exact local source revision, resolved launch configuration, model, data, resource layout, and artifact destinations that produced the job.
- Classify the run as completed, failed, cancelled, or indeterminate. For a failure, retain the first causal error as well as the terminal controller state.

## 2. Preserve durable evidence

Collect or verify access to the artifacts defined by the resolved launch:

- training metrics, trainer logs, and relevant worker logs;
- checkpoints and export metadata;
- agentic trials, rewards, verifier outcomes, and trace/literal evidence when configured;
- the resolved launch command and configuration.

Use the existing artifact backend and experiment destination. Copy only the evidence required for diagnosis or archival, retain its provenance, and verify transfers before considering the source disposable.

## 3. Validate a candidate result

Before calling a checkpoint or export usable, verify that it is complete for the requested next action:

- checkpoint shards and metadata agree on the same completed step;
- tokenizer/configuration files match the intended model export;
- metrics and trials cover the claimed interval without unexplained terminal errors;
- agentic runs retain the trial-level evidence required to interpret rewards.

Select an artifact only by an experiment-defined criterion. If no criterion is supplied, report the complete candidates and leave selection to the user rather than silently choosing a "best" one.

## 4. Hand off or publish only with authority

- Publish or register an artifact only when the requested scope authorizes it. Verify the intended destination, visibility, provenance, and secret-free metadata first.
- Never delete raw checkpoints, trials, remote objects, or database records as a side effect of cleanup.
- For a failed run, distinguish infrastructure, configuration, model/training, and agent/verifier failures. State whether a resume is safe and the smallest corrective action required.

## Completion record

Return the terminal state, source revision, resolved configuration, preserved artifact locations, validation result, selected artifact if authorized, and any remaining action. Keep the record factual and tied only to the observed run and its declared experiment policy.
