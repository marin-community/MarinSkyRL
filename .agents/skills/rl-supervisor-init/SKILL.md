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

- Do not create worktrees or edit repository source, configuration, or skills. You may update the
  selected experiments' mutable STATE.md operations records, but must not commit those edits.
- Do not commit, push, open or update a PR, merge code, or implement a recommended fix.
- Load credentials only through approved secret mechanisms; never print or persist their values.
- You may inspect repository state and operate Iris jobs only within the authority granted by the
  user or campaign policy.
- Hand source and configuration defects to an implementation role with evidence, expected behavior,
  affected component, and a proposed regression test.
- Never patch a live pod or remote checkout.
- Never edit a POLICY.md document without direct user authorization.
- Never autonomously override a guideline in POLICY.md without direct user authorization. e.g., if a POLICY.md says "always keep 3 jobs in flight," do not launch 2 or 4 jobs.
- Treat each experiment's STATE.md as its mutable operations record. Update it within the selected
  scope and remove stale information as soon as it is discovered.
- Scrutinize subagent reports against controller state and durable artifacts before acting.
- Keep mutable job state out of recurring prompts. Make each recurrence reread the experiment
  records and current Iris state.

## Supervision loop

1. Discover the repository root and read the relevant `.agents/ops/` runbooks.
2. Use an explicit experiment list when provided. Otherwise discover active RL experiment records
   through the current operations conventions. Read each selected POLICY.md before acting.
3. Run `scripts/iris/watch_coreweave_rl.py` for the initial survey. If recurring execution is
   available, schedule a three-hour survey that links the selected experiment records and rereads
   them on every recurrence.
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
