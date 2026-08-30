# RL supervisor recurring prompt

Use this prompt for every recurring `rl-supervisor-init` sweep. The prompt contains procedure only.
Experiment paths enter through the skill's `experiment_dirs` argument and must remain separate from
the prompt text.

## Scope binding

Invoke `rl-supervisor-init` with one or more explicit, absolute experiment directories. That argument
list is the complete supervision scope. Each directory must contain `POLICY.md` and `STATE.md`; it may
also contain `TRACKER.md` and an `artifacts/` directory.

Bind the unchanged `experiment_dirs` arguments to every recurrence through the scheduler's retained
task context or argument mechanism. Do not interpolate paths, job IDs, status, reminders, or campaign
decisions into the prompt. If the recurrence mechanism cannot retain the skill arguments separately,
do not schedule it; report the setup limitation to the user.

## Exact prompt

Copy the text inside this block verbatim. Do not prepend, append, paraphrase, or specialize it.

```text
Run one RL supervision sweep for every experiment directory passed to the rl-supervisor-init skill. Do not add, remove, or infer experiments.

For each experiment:
1. Reread POLICY.md and STATE.md. Read TRACKER.md when present. Inventory artifacts/ when present; you may edit its contents, including scripts and configuration, but never repository source, configuration, or skills.
2. Compute the SHA-256 digest of POLICY.md and compare it with the `Policy SHA-256: <digest>` entry in STATE.md. If no digest is recorded, record the current digest as the baseline without claiming that policy changed. If it changed, apply the current policy and record the new digest and operational consequences in STATE.md. Never edit POLICY.md.
3. Follow .agents/ops/watch-coreweave-rl.md. Inspect and refresh the canonical local syncdown first, then reconcile every tracked job against the authoritative controller on the cluster where it runs. Use live log or process probes only for missing, contradictory, or ephemeral evidence.
4. Update STATE.md with current observations, actions, decisions, evidence locations, and unresolved questions. Update TRACKER.md when present so its job order, eligibility, outcomes, and next executable work agree with POLICY.md, STATE.md, and cluster state. Remove or correct stale claims instead of appending contradictions.
5. Investigate every ambiguous job with the rl-job-health-deep-dive skill. Check its recommendation against controller state and durable artifacts before acting.
6. Perform only lifecycle actions authorized by the user or POLICY.md. Preserve evidence that would disappear before any authorized stop.
7. For a source, configuration, launcher, or tooling defect, write a concise escalation report under /Users/benjaminfeuer/Documents/agent_logs with the evidence, affected component, expected behavior, and a regression test. Do not patch repository code in the supervisor role.
8. Report the sweep to the user. Use GitHub-flavored Markdown pipe tables, identify every item requiring user action, and state explicitly when no user action is required.
```

## Maintenance

The supervisor role must never edit this file or alter the prompt while supervising experiments.
Change the canonical prompt only through a separate implementation workflow that reviews the ops doc
and supervisor skill together.
