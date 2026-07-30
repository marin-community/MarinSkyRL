---
name: rl-job-health-deep-dive
description: >-
  Diagnose one live or suspect Iris RL job and return an evidence-backed KILL, NO-KILL, or ERROR
  recommendation. Use when ordinary state and metric checks cannot distinguish progress from a
  silent stall, or when validating a new model, geometry, configuration, or image. This is a
  read-only repository role: it never edits code, commits, pushes, opens PRs, or kills jobs.
---

# Deep-dive one Iris RL job

Inspect one job and return a recommendation to its supervisor. Read
`.agents/ops/coreweave.md` for access and capture procedures and
`.agents/ops/rl-diagnostics.md` for signal interpretation before probing.

## Role boundary

- Do not create a worktree or edit repository files, configurations, skills, or ops documents.
- Do not commit, push, open or update a PR, or apply a code fix.
- Do not kill, stop, relaunch, or mutate a job. The supervisor owns job actions.
- You may perform read-only repository inspection and non-mutating live probes.
- Express any required source or configuration change as a concrete handoff: evidence, likely cause,
  affected path or component, proposed behavior, and a test that would prove the fix.

If an ops document is stale or incomplete, report the discrepancy. Do not repair it in this role.

## Capture before analysis

Sync the durable job logs and analyze the local evidence bundle. Use live probes only for evidence
that disappears with the job, such as per-rank GPU state or process stacks. Preserve exact commands,
timestamps, file paths, and representative lines so another operator can reproduce the verdict.

Return `ERROR` when required evidence cannot be obtained or authoritative signals remain
unreconciled. Missing evidence is not a conservative `NO-KILL`.

## Required gates

1. **Restarts:** determine whether attempts repeat the same failure or recovered from a transient.
2. **Liveness:** pair authoritative controller state with the newest phase or step advancement.
3. **Resources:** separate policy and inference ranks; interpret engine saturation using the tuple in
   the diagnostics runbook, not point-in-time utilization.
4. **Rollouts:** inspect actual rewards, outputs, verifier results, and exception artifacts.
5. **Dynamics:** extract reward, entropy, TIS, and phase-time series over the minimum useful window
   specified in the diagnostics runbook.
6. **Optional duty cycle:** measure bounded per-trial timings only when attribution between
   generation, tools, and sandbox lifecycle changes the recommendation.

Each of the four core gates is `PASS`, `FAIL`, or `ERROR`. Report restarts separately and include
the optional duty-cycle evidence when used. Quote the evidence supporting every classification.

## Verdict rules

- `KILL`: a hard failure is established and has no transient or benign explanation. A stall verdict
  must include live process evidence captured while the job still exists.
- `NO-KILL`: the gates pass, or an observed anomaly has a supported transient or initialization
  explanation. State the precise signal that would change the verdict.
- `ERROR`: a required gate lacks evidence, a probe failed, or sources disagree without resolution.

For learning-quality concerns, require the configured minimum observation window, agreement between
trainer and trial evidence, a named mechanism, and evidence that the run is not already recovering.
A plateau alone is not a kill condition. Read any artifact-retention threshold from the campaign
record and surface its consequence without weakening a hard-failure verdict.

## Report format

```text
RL-JOB-HEALTH — <job-id> — captured <timestamp> at <evidence-dir>

VERDICT: KILL | NO-KILL | ERROR
CONFIDENCE: high | medium | low
RESTARTS: <count and interpretation>

LIVENESS:  PASS | FAIL | ERROR — <quoted evidence>
RESOURCES: PASS | FAIL | ERROR — <quoted evidence>
ROLLOUTS:  PASS | FAIL | ERROR — <quoted evidence>
DYNAMICS:  PASS | FAIL | ERROR — <metric window and phase table>

REASONING: <load-bearing evidence and mechanism>
NEXT ACTION: <supervisor job action or implementation handoff>
FLIP SIGNAL: <specific condition that would change this recommendation>
```
