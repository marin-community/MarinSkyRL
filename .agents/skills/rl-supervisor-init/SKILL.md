---
name: rl-supervisor-init
description: >-
  Initialize and run a focused RL operations session across Iris and explicitly selected supported
  backends: inventory jobs, dispatch diagnosis, perform authorized lifecycle actions, and maintain a
  reproducible handoff for an explicit list of experiment directories. This is a no-commit role that
  never edits repository code or configuration and never opens or updates pull requests.
---

# Initialize an RL supervisor

Accept `experiment_dirs` as a required, nonempty list of explicit absolute experiment directories.
Treat this argument as the complete, immutable scope for the supervision session and every
recurrence. Do not discover, add, remove, or substitute experiments. Ask the user for the scope when
the argument is missing or ambiguous.

Validate that every directory contains `POLICY.md` and `STATE.md`. Read `TRACKER.md` when present.
Treat `artifacts/` as optional experiment material that may include untracked scripts, resolved
requests, and reports; you have authority to edit content you find there.

Read the current launcher interface, selected configuration, each experiment record, and
`.agents/ops/watch-coreweave-rl.md` at the start of every session. Read the additional ops runbooks it
routes to when the selected backend or evidence requires them.

Within each selected experiment, `POLICY.md` defines operational authority and invariants;
`STATE.md` records mutable observations, actions, decisions, and the last observed policy digest;
`TRACKER.md`, when present, records ordered or eligible work and its outcomes.

## Role boundary

- Do not create worktrees or edit repository source, configuration, skills.
  You may update the selected experiments' mutable `STATE.md` and `TRACKER.md` records and write
  escalation reports to the location defined by the recurring-prompt runbook, but must not commit
  them.
- Do not commit, push, open or update a PR, merge code, or implement a recommended fix.
- Load credentials only through approved secret mechanisms; never print or persist their values.
- You may inspect repository state and operate jobs only within the authority granted by the user or
  the experiment's `POLICY.md` and the selected backend's ops runbook.
- Hand source defects to an implementation role with evidence, expected behavior,
  affected component, and a proposed regression test.
- Never patch a live pod or remote checkout.
- Never edit a POLICY.md document without direct user authorization.
- Follow each POLICY.md directive exactly. Altering a declared target or invariant requires direct
  user authorization.
- Treat each experiment's `STATE.md` as its mutable operations record. Update it within the selected
  scope and remove stale information as soon as it is discovered. Keep `TRACKER.md`, when present,
  consistent with current policy, state, and cluster evidence.
- Scrutinize subagent reports against controller state and durable artifacts before acting.
- Never put experiment paths or mutable job state in the recurring prompt. You may list the user-passed `experiment_dirs` as an exception to this rule.
- Routine ingress and egress within a selected CoreWeave job scope do not require confirmation for
  each action. Large or cross-region transfers remain subject to repository policy.

## Supervision loop

1. Discover the repository root and read the relevant `.agents/ops/` runbooks.
2. Validate the required `experiment_dirs` argument. Read each selected `POLICY.md` before acting;
   compare its SHA-256 digest with the value recorded in `STATE.md`, and record a baseline without
   claiming a change when no prior digest exists.
3. Follow `.agents/ops/watch-coreweave-rl.md` for the initial survey. Inspect and, when needed,
   refresh the canonical local syncdown before querying a cluster. Use live probes only for missing,
   contradictory, or ephemeral evidence.
4. Incorporate user decisions received since the prior survey into each experiment's STATE.md.
5. Diagnose every suspect job with a separate `rl-job-health-deep-dive` subagent. Wait for all
   reports, check them against current evidence, and monitor stalled probes at a bounded interval.
6. Update each STATE.md with every job's status, evidence, and unresolved decision.
7. Separate proposed actions into:
   - escalation actions outside the supervisor's authority;
   - management actions explicitly authorized by the user or experiment POLICY.md.
8. Record each escalation in the configured operations log location with evidence and recommended
   next steps.
9. Execute authorized management actions in this order: KILL, CLEANUP, then LAUNCH. Update STATE.md
   after each action.
10. Report the completed sweep and identify decisions that still require the user.

## Recurring execution

Read `.agents/ops/rl-supervisor-cron-prompt.md` before scheduling recurrence. Always use the exact
prompt block from that document. Never rewrite, summarize, extend, or interpolate values into it.
Bind the original `experiment_dirs` argument separately through the recurrence mechanism's retained
context or arguments.

Schedule the survey every three hours when recurring execution is available. If the scheduler cannot
retain the experiment arguments without changing the prompt, do not create the recurrence and report
the limitation to the user. The supervisor role must never edit the canonical cron-prompt ops doc;
prompt changes require a separate implementation workflow.

## User-facing tables

Render every table in a supervisor response as a GitHub-flavored Markdown pipe table. Do not paste
the watcher's Unicode terminal table or place a table in a code fence. Use the exact RL status
columns and Markdown example in `.agents/ops/watch-coreweave-rl.md` for fleet updates. Keep cells
compact, use `—` for unavailable values, and put evidence or caveats that do not fit in a cell after
the table.

## Dispatch and authority

| Need | Procedure | Authority boundary |
|---|---|---|
| Diagnose one job | `rl-job-health-deep-dive` | recommendation only; no mutation |
| Launch agentic RL | `rl-agentic-launch-iris` | dry-run first; submit only when authorized |
| Launch standard RL | `rl-standard-launch-iris` | dry-run first; submit only when authorized |
| Preserve agentic output | `rl-agentic-job-cleanup` | preservation transfers are allowed; reclamation follows cleanup authority |
| Preserve standard output | `rl-standard-job-cleanup` | preservation transfers are allowed; reclamation follows cleanup authority |
| Rebuild an image | hand off to an implementation role using `build-gpu-rl-image-iris` | never performed by the supervisor role |

Cancellation, deletion, publication, registration, credential rotation, and database mutation each
require explicit authority unless the experiment's POLICY.md grants that exact action. Capture evidence
that disappears at termination before any authorized stop.
