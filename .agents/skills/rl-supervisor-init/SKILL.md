---
name: rl-supervisor-init
description: >-
  Initialize and run a focused Iris RL operations session: inventory jobs, dispatch diagnosis,
  perform authorized lifecycle actions, and maintain a reproducible handoff. This is a no-commit
  role that never edits repository code or configuration and never opens or updates pull requests.
---

# Initialize an Iris RL supervisor

Establish current facts for the jobs and campaign the user placed in scope. Read the current
launcher interface, selected configuration, campaign record, and relevant ops runbooks at the start
of every session.

## Role boundary

- Do not create worktrees or edit repository source, configuration, skills, or ops documents.
- Do not commit, push, open or update a PR, merge code, or implement a recommended fix.
- You may inspect repository state and operate Iris jobs only within the authority granted by the
  user or campaign policy.
- Hand source and configuration defects to an implementation role with evidence, expected behavior,
  affected component, and a proposed regression test.
- Never patch a live pod or remote checkout.

## Establish the operating state

1. Confirm the local source revision and whether the checkout is clean. Treat it as evidence, not an
   editable workspace.
2. Read the selected RL configuration and current launcher help. Resolve defaults dynamically.
3. Load credentials only through approved secret mechanisms; never print or persist their values.
4. Query the requested jobs and classify each as starting, productive, stalled, terminal, or
   indeterminate from controller state and durable artifacts.
5. Classify agentic versus standard RL from resolved configuration, not job names.

## Dispatch and authority

| Need | Procedure | Authority boundary |
|---|---|---|
| Diagnose one job | `rl-job-health-deep-dive` | recommendation only; no mutation |
| Launch agentic RL | `rl-agentic-launch-iris` | dry-run first; submit only when authorized |
| Launch standard RL | `rl-standard-launch-iris` | dry-run first; submit only when authorized |
| Preserve agentic output | `rl-agentic-job-cleanup` | publish/register only as its authority section permits |
| Preserve standard output | `rl-standard-job-cleanup` | publication requires explicit authority |
| Rebuild an image | hand off to an implementation role using `build-gpu-rl-image-iris` | never performed by the supervisor role |

Cancellation, deletion, publication, registration, credential rotation, and database mutation each
require explicit authority unless the campaign record grants that exact action. Capture evidence
that disappears at termination before any authorized stop.

## Recurring supervision

A recurring prompt must be self-contained or name the live records it rereads. Include:

1. job scope and exact operational authority;
2. campaign policy and state records as the source of mutable parameters;
3. inventory query and state interpretation;
4. terminal-job preservation before live-job diagnosis;
5. per-job probes and previously declared flip conditions;
6. launch gates and refill policy;
7. escalation and evidence-preservation rules;
8. reporting destination and a termination condition;
9. a stable-path status artifact when the campaign needs a dashboard.

Do not inline mutable quotas, thresholds, run identifiers, image revisions, or current capacity in a
durable prompt when a campaign record can own them.

## Handoff

Record observed job identifiers, source revisions, resolved configurations, artifact locations,
evidence links, actions taken, and pending decisions in the campaign record. For every recommended
code/configuration change, state that an implementation agent must reproduce, test, review, and land
it through the repository workflow.
