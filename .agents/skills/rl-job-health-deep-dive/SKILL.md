---
name: rl-job-health-deep-dive
description: >-
  Deep single-RL-job health probe on the CoreWeave clusters → one evidence-backed KILL / NO-KILL /
  ERROR recommendation for the supervisor. Use on any RL job in a new or untested setting (new
  config, geometry, model, or image; first launch after a code change) and on any running job that
  looks starved or wedged — wherever a state poll plus table metrics cannot separate "progressing"
  from "silently dead." The prober never kills; the supervisor owns the kill. Cluster access and
  capture commands live in .agents/ops/coreweave.md; log/metric semantics live in
  .agents/ops/rl-diagnostics.md — read those, do not restate them.
---

# rl-job-health-deep-dive

Probe one RL job hard and return one recommendation. You are a subagent: you never execute a
kill. When genuinely uncertain, prefer NO-KILL and escalate — a wrongly killed healthy run
wastes a full bring-up; a wrongly kept dead one wastes one sweep. But if you could not get the
evidence, the answer is ERROR, not a hedged NO-KILL.

**Facts live in the ops docs, not here.** Read `.agents/ops/coreweave.md` (access, state poll,
log capture, GPU poll, py-spy, guardrails) and `.agents/ops/rl-diagnostics.md` (what the logs
and metrics mean) before probing. If a needed fact is missing from those docs, add it there and
point — never inline it here.

## Step 0 — capture first, analyze from files

Before any gate: sync the job's logs local with `infra/sync_rl_logs.py` (command in
`coreweave.md` §Log capture) and do all analysis from the local files. Live probes are reserved
for what files cannot give: the GPU poll and a py-spy on a still-running suspect (both die with
the job — capture them before recommending any kill). Hand the supervisor exact local file
paths plus quoted lines so the verdict can be confirmed from the same files.

## The contract — evidence or ERROR

Return `VERDICT: ERROR` whenever you could not obtain required evidence: a tool failed, a log
could not be fetched, policy and engine GPUs could not be separated, or two authoritative
signals disagree unreconciled. Report the exact command, its exact failure output, what is
missing, and what you did establish. Never emit KILL/NO-KILL on missing evidence, and never
default to NO-KILL as a hedge.

| Gate | A PASS/FAIL requires you to have read and quoted | else the gate is |
|---|---|---|
| A liveness | the authoritative state-poll line and the newest phase-timer/step line with its timestamp | ERROR |
| B resources | per-rank GPU util with policy ranks separated from engine ranks, and the engine subscription line (Running vs Waiting vs cap) | ERROR |
| C rollouts | actual reward values / trial exception files you opened, not counts you assumed | ERROR |
| D dynamics | the per-step metric series (reward, entropy, TIS, phase timers) you extracted from the synced finelog over a ≥10-step window | ERROR |

## Probe order

1. **Restart-burn check** (cheap; `rl-diagnostics.md` §Restart-burn): restarts burned, and is
   it the same failure every attempt?
2. **Gate A — liveness.** Authoritative state poll + log freshness against the run's own
   cadence + wedge/death signatures. On a multi-mesh job, clear the colocated-engine deception
   checks (`rl-diagnostics.md`) before calling PASS. On any death or wedge, read both
   `finelog.log` and `ray_session_logs/` — the root cause hides in one of them, and which one
   varies.
3. **Gate B — resources.** Live GPU poll, policy ranks separated from engine ranks; engine
   subscription read via the saturation tuple, not SM-util. For suspected starvation, take the
   pipeline measurements named in `rl-diagnostics.md` §Engine saturation while the job is
   alive; an unmeasured starvation claim is ERROR-quality.
4. **Gate C — rollout quality.** Open actual trial artifacts: rewards, turns, coherence,
   verifier behavior. Match against the known signatures in `rl-diagnostics.md` before
   inventing a new theory.
5. **Gate D — training dynamics.** State, GPUs, and rollouts can all be green while the run
   quietly stops learning or bleeds throughput. Extract four series from the synced finelog
   (and wandb when configured) and read each against `rl-diagnostics.md` §Training dynamics:
   reward, entropy, the TIS family, and a per-phase Timer table. Report the phase table and
   the metric values, not an impression. A drifting step time with no pathology is a tuning
   note, not a KILL.
6. **Optional deep probe — per-trial duty cycle.** Only when you must decide whether trial
   throughput is capped by generation vs sandbox lifecycle vs tool exec: aggregate per-trial
   timing from trial artifacts in bounded samples (`rl-diagnostics.md` §Per-trial duty
   cycle), and never assert "sandbox churn" without those numbers.

## Deliver one recommendation

```
RL-JOB-HEALTH — <job_id>  (<model>, <geometry>, <stage>)   captured: <dir>

VERDICT: KILL | NO-KILL | ERROR          confidence: high|medium|low
Evidence I actually read: <quoted state-poll line; policy-vs-engine util split; engine
  subscription counts; reward values / exception files; the metric window. A blank row means
  that gate is ERROR.>
Restarts: <burned/max — none | same failure each attempt | transient, recovered>

Gate A (liveness):  PASS|FAIL|ERROR — <evidence>
Gate B (resources): PASS|FAIL|ERROR — <evidence>
Gate C (rollouts):  PASS|FAIL|ERROR — <evidence>
Gate D (dynamics):  PASS|FAIL|ERROR — <reward trend over the window; entropy; TIS reads;
                    per-phase Timer table and any bubble>

REASONING: <2–4 sentences; the load-bearing evidence, especially for whatever is new/untested>
NEXT STEPS: <KILL: root cause + concrete fix + relaunch-or-hold.
             NO-KILL: what to watch next tick + the signal that would flip the verdict.
             ERROR: what to fix so the next probe gets the evidence.>
```

Verdict rules:
- **KILL** — a gate hard-fails with no transient or benign explanation, and you hold the
  evidence. A starvation/wedge KILL must include the live py-spy captured before the kill.
- **KILL (deterministically doomed)** — restarts repeat the same failure every attempt; state
  the count, the traceback, and the fix that must land first.
- **NO-KILL** — all gates pass, or the only failures have a legitimate transient or
  early-bring-up explanation. Say what you are waiting on.
- **ERROR** — required evidence unobtainable. Never launder it into NO-KILL.

## The asymmetry on learning-quality verdicts

The costs here are not symmetric, and the recommendation should reflect that. A wrongly killed
healthy run forfeits every banked step and costs a full bring-up to replace. A wrongly kept
unhealthy run costs one sweep of one arm, and the next probe catches it. **On anything that turns
on training dynamics rather than a hard fault, be permissive: when the evidence is mixed, the
answer is NO-KILL with a sharpened watch, not KILL.**

A hard fault — a wedge, a dead rank, a crash loop, a job producing nothing — is different. Those
are cheap to establish and expensive to leave running. This section is about the other kind.

**Do not recommend KILL for "it does not look like it is learning" unless all of these hold:**

1. **The window clears the ≥10-step bar.** Reward and entropy on these runs are noisy enough that
   short windows routinely show trends that reverse. Runs have recovered after a five-step decline
   and after an entropy trough that looked like the start of a collapse.
2. **A trial-level measurement agrees with the trainer-side one.** They disagree in both
   directions: the trainer lags live generation by roughly `staleness_mean` × cycle, and a
   trial-level window can be dominated by a burst. One source alone is not enough to end a run.
3. **The mechanism is named.** "Reward is falling" is an observation. "Reward is falling because
   the policy is doing X, evidenced by Y" is a finding. Without the mechanism you cannot tell
   degradation from a hard patch of the curriculum.
4. **It is not recovering.** Check whether the metric has already turned before recommending
   anything.

If some but not all hold, say exactly which, recommend NO-KILL, and propose the specific
measurement that would settle it next tick. A verdict of "NO-KILL, and here is the one number that
would change my mind" is more useful to the supervisor than a confident KILL on four steps.

**Flat is not dying.** A run whose reward has stopped improving is not the same as one that is
collapsing, and a plateau alone is never sufficient grounds. Plateaus resolve in both directions.

**Say so when you are near the line.** If you would have recommended KILL on marginally different
numbers, state that plainly along with the numbers. The supervisor holds the kill authority and can
weigh cost against confidence; a probe that hides its uncertainty removes that choice.

The supervisor executes any teardown and relaunch. Diagnosis fixes go through the normal
worktree → PR flow; never hand-patch a cluster.
