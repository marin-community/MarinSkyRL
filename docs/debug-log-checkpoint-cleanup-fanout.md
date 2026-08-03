# Debugging log for the checkpoint-cleanup fan-out

Stop `_cleanup_old_checkpoints` from killing TaskTrove arms at their first
checkpoint when `max_ckpts_to_keep` is negative (the default).

## Initial status

Four TaskTrove arms FAILED exit `1:0` within 23 minutes, each on a different
node group, each immediately after banking its first checkpoint
(`global_step_3`). Identical signature in all four driver logs: 3 ×
`Task cleanup_old_checkpoints failed` and 4 × `WorkerCrashedError`. The driver
traceback bottomed out in `RayPPOTrainer._cleanup_old_checkpoints`. No OOM, no
CUDA OOM, host RAM idle immediately before the crash, and inode quota clear.

Source escalation:
`2026-08-03_escalation-checkpoint-cleanup-fanout-kills-every-tasktrove-arm.md`.

## Established call chain

`save_checkpoints` (`skyrl_train/trainer.py`) logs a successful save, then under
a `Timer` calls `self._cleanup_old_checkpoints()`. That method unconditionally
calls `run_on_each_node(self._node_ids, cleanup_old_checkpoints, ...)` followed
by a driver-side `cleanup_old_checkpoints(...)`.

`run_on_each_node` (`skyrl_train/utils/trainer_utils.py:62`) leases a fresh
0.25-CPU Ray worker on each node with `NodeAffinitySchedulingStrategy(soft=False)`
and blocks on `ray.get(refs)`.

## Hypothesis 1 — a GPFS delete race interrupts the payload

Refuted. The dead runs resolved `max_ckpts_to_keep: -1` (the
`ppo_base_config.yaml:211` default; the campaign configs never override it). The
payload's first statement is `if max_checkpoints < 0: return`, so no
`list_checkpoint_dirs`, no `io.remove`, and no filesystem access occurs. The
surviving `global_step_3` is therefore not an interrupted delete; nothing was
ever deleted. `max_ckpts_to_keep: 0` is also impossible — `utils/utils.py:531`
rejects it at config validation.

## Established mechanism — the dispatch, not the payload

Because the payload is a no-op at this configuration, the only thing that can
kill the workers is the dispatch itself. At checkpoint time the trainer requests
a fresh 0.25-CPU Ray worker on every node with hard node affinity. Jupiter's
GH200 nodes are 4-GPU with the agent harness resident, so their CPU is fully
committed; the lease fails or the worker is killed during lease, Ray reports
`SYSTEM_ERROR` on three node IPs, the cleanup task retries 2 → 1 → 0, and the
`WorkerCrashedError` propagates through `ray.get(refs)` into an unguarded
`save_checkpoints`, killing the driver.

CoreWeave/Iris arms hit the same fan-out but their nodes have CPU headroom
(48 cpu/node requested against larger nodes), so the lease succeeds.

## Changes to make

`skyrl_train/trainer.py`, `RayPPOTrainer._cleanup_old_checkpoints`:

1. Return before any Ray work when `max_ckpts_to_keep < 0`. The common path
   (the default config) stops taking a multi-node dependency entirely, and the
   driver-side payload call is skipped too (it is a no-op at this value).
2. Isolate the fan-out failure at the call site. Cleanup runs only after a
   successful checkpoint save, so it is best-effort housekeeping and must not be
   able to kill a run whose checkpoint is already on disk. A `WorkerCrashedError`
   from a saturated node is logged, not propagated.
3. Drop the stale "it's ok because it's idempotent" comment, which describes a
   property irrelevant to the failure, and replace it with a comment that names
   the driver-side call's actual role (sufficient for a shared `ckpt_path`).

## Regression tests

`skyrl-train/tests/cpu/test_checkpoint_cleanup.py`, against a bare
`RayPPOTrainer` (bypass `__init__`, following `test_resume_overshoot.py`):

1. `max_ckpts_to_keep = -1` with a multi-node `_node_ids` list dispatches no Ray
   task. `run_on_each_node` (the Ray lease boundary) is patched and asserted not
   called; this is the only faithful check, because once the fan-out is wrapped
   in `except RayError` a "does it raise" check passes whether or not the early
   return exists.
2. When the fan-out raises `WorkerCrashedError`, the driver-side cleanup pass
   still runs and removes old checkpoints (asserted on a real temp dir), so a
   shared `ckpt_path` is still pruned.

A unit test on `cleanup_old_checkpoints` alone passes today and misses both,
because the defect is in the dispatch and its error handling, not the payload.

## Results

`RayPPOTrainer._cleanup_old_checkpoints` now returns before any Ray work when
`max_ckpts_to_keep < 0`, wraps the per-node fan-out in `except RayError` so a
worker lease failure or node loss is logged rather than propagated, and keeps
the driver-side cleanup call so a shared `ckpt_path` is still pruned. The stale
"it's ok because it's idempotent" comment was replaced.

Regression tests in `skyrl-train/tests/cpu/test_checkpoint_cleanup.py`:
no dispatch when cleanup is disabled, and the driver-side pass still removes
old checkpoints after a `WorkerCrashedError`. Both fail against the pre-fix
code and pass after it; the full `tests/cpu/` suite is green and
`infra/pre-commit.py` is clean.

## Future work

- [ ] Consider whether `run_on_each_node` should default to `soft=True` so a
      saturated node falls back to placement instead of failing hard; out of
      scope for this fix because the early return plus call-site isolation make
      it moot for the default configuration.
