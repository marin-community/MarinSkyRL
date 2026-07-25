---
name: rl-supervisor-init
description: Initialize a focused MarinSkyRL supervisor session for Iris-only RL operations. Use at the start of an operational session before monitoring, diagnosing, launching, or recovering Iris RL jobs.
---

# Initialize an Iris RL Supervisor Session

Use this workflow to establish current, reproducible facts for MarinSkyRL training on Iris. It is intentionally limited to this repository, its companion Marin checkout, and the Iris jobs placed in scope by the user.

## 1. Establish source-of-truth state

- Inspect the local MarinSkyRL checkout, its active branch, and uncommitted changes. Treat local committed source and configuration as the only launchable ground truth.
- Confirm that the companion Marin checkout and the current Iris launcher/configuration interface are available.
- Read the selected RL configuration and current launcher help before making any operational decision. Resolve dynamic defaults from those sources rather than from this skill.
- Load credentials only through the approved environment/secret mechanism. Do not display, copy, or persist secret values.

## 2. Inventory the requested Iris scope

Query the current Iris control plane for the user's requested jobs and time window. For each job, identify its state, creation/update time, resolved configuration or launch metadata, resource layout, and durable artifact locations.

Classify each job from current evidence:

- **starting**: admitted but workers, serving, or the trainer are still initializing;
- **productive**: trials or batches are completing and training metrics/steps are advancing;
- **stalled**: no meaningful progress inside the configured expectation, with evidence of the blocking stage;
- **terminal**: completed, failed, or cancelled, with the first causal error retained when available;
- **indeterminate**: required job metadata or artifacts are unavailable.

Classify agentic versus standard RL from the resolved configuration, not the job name.

## 3. Report and decide safely

Report the current state, objective evidence, and next safe action for every job in scope. Use the current launcher/configuration for resource and retry expectations; do not embed capacity assumptions or campaign limits in the report.

- Launches and relaunches must use the relevant Iris RL skill and a reviewed dry run.
- Diagnoses must preserve first-cause evidence and distinguish framework, configuration, infrastructure, training, and agent/verifier faults.
- Cancellation, deletion, publication, credential rotation, and database mutation require explicit user authority.
- Never repair a job by modifying a live pod, remote checkout, or remote configuration. Make local changes, validate them, and submit a new attempt.

## 4. Leave a reproducible handoff

For work that continues beyond the current session, record only the observed job identifiers, source revisions, resolved configurations, artifact locations, evidence links, and pending decisions in the user-owned experiment record. Do not turn the skill itself into a campaign log or a repository of mutable operational state.
