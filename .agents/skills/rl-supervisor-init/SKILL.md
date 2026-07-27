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

## 4. Know what each companion skill does — and what it deliberately will not do

Dispatch work to the companion skills rather than improvising, and brief the subagent on the
skill's **limits** as well as its task. Most supervision mistakes come from assuming a skill
covers a step it never claimed.

| Need | Skill | What it will NOT do |
|---|---|---|
| Probe one live job's health | `rl-job-health-deep-dive` | never kills; returns a recommendation only |
| Wind down an agentic run | `rl-agentic-job-cleanup` | **does not publish, register, or delete** |
| Wind down a non-agentic run | `rl-standard-job-cleanup` | same |
| Launch or relaunch agentic | `rl-agentic-launch-iris` | — |
| Launch or relaunch standard | `rl-standard-launch-iris` | — |
| Rebuild the container | `build-gpu-rl-image-iris` | — |

**The cleanup skills preserve and validate; they do not publish.** Publication, registration, and
deletion each require explicit user authority, by design. A supervisor who assumes "cleanup ran,
therefore the model is on the Hub" will be wrong. If the campaign expects a published artifact at
the end of every run, that is a separate authorized step the supervisor must schedule and confirm —
name it in the cron prompt, and verify the destination actually received the artifact rather than
trusting an exit code.

**Verify publishability before promising it.** A cleanup can only upload an artifact that exists.
Confirm the finished run actually wrote the format the destination needs; a training checkpoint and
a publishable model are not the same object, and a run can exit zero having produced only the
former. Report "nothing to publish" as a finding, not as a completed publication.

## 5. Drafting a cron prompt for sweep monitoring

A recurring supervision prompt is re-read by an agent with **no memory of previous ticks**. Every
fact it needs must be in the prompt or reachable from a document the prompt names. Write it to
survive that.

Include, in order:

1. **Role and authority.** What the agent may do without asking — typically kill and relaunch
   within the campaign — and what it may not: publish, register, delete artifacts, or touch
   anything outside the named scope.
2. **The experiment record as the authority.** Name the policy/state/tracker documents and say to
   re-read the policy each tick, because it changes. The prompt should carry procedure; the
   campaign's current parameters live in the record.
3. **Inventory, with the query and the state encoding.** Give the exact query and spell out the
   numeric states. Warn about query forms that return an empty set on malformed input, so an agent
   does not read "no rows" as "no jobs".
4. **Terminal jobs before live ones.** A finished or dead run is evidence that decays; a live one
   can be probed next tick. Say which cleanup skill applies to which run type, and require the
   outcome be written to the tracker.
5. **Per-job probes**, dispatched to the health skill, carrying **each job's prespecified flip
   rule from the previous tick** — the threshold, the window, and the metric — with an instruction
   to apply it literally rather than re-derive it. A rule invented fresh each tick is not a rule.
6. **Any screen the campaign gates launches on**, with its metric, its bounds, and where to read
   it from. State the source explicitly: under async generation the trainer's copy of a metric can
   lag the live rollouts by `staleness × cycle`, so a screen read from trainer metrics can pass an
   arm that has already failed.
7. **Refill policy.** The quota, what counts toward it, and which harness/config is current.
   Queued and pending work counts against the quota; a job whose row exists but whose tasks are
   still gated has not started.
8. **Escalation limits.** Never stop a job showing a signal that is not yet understood — a live
   job is the reproduction, and an unrelated but valid reason to stop it still destroys the
   evidence. Require live-only evidence to be captured first when a stop is unavoidable. Require
   the supervisor to verify a probe's load-bearing claim before acting on a kill.
9. **Reporting and hygiene.** Where the log goes, and that checkouts return to their canonical
   branch with no residue — probes sometimes sync logs into a repo root.
10. **A termination condition**, so the schedule ends itself when the campaign is done rather than
    firing indefinitely.

Two failure modes worth designing against: a prompt that names no authority forces the agent to
ask permission every tick and the schedule stops being autonomous; a prompt that carries campaign
parameters inline goes stale the first time the policy changes, and the agent then acts on numbers
the owner has already revised.

## 6. Leave a reproducible handoff

For work that continues beyond the current session, record only the observed job identifiers, source revisions, resolved configurations, artifact locations, evidence links, and pending decisions in the user-owned experiment record. Do not turn the skill itself into a campaign log or a repository of mutable operational state.
