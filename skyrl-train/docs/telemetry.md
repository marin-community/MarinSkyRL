# MarinSkyRL telemetry

Install `skyrl-train[telemetry]` to export driver and trainer lifecycle, policy
steps, generated rollouts, samples and tokens, exclusive rollout or inference
wait and train-step durations, and fully async rollout-buffer occupancy through
`rigging.telemetry`. The same extra lets each Iris controller forward a bounded
allowlist of its local Ray scheduler, logical CPU/GPU, placement-group and object
store snapshots. A rollout is one completed trajectory; a sample is one generated
response segment, so step-wise training counts only terminal segments as
rollouts. Export is inert without a telemetry endpoint, run id, and execution uid.
`cloud/iris/telemetry_env.py` resolves them inside the Iris task, and the task
runtime exports them before Ray starts so its actors inherit them.
`SKYRL_EXECUTION_UID` can override the execution identity; otherwise each process
uses its node-local `IRIS_ATTEMPT_UID`. The service is fixed to `marinskyrl`;
`SKYRL_SERVING_JOB_ID` optionally joins a centralized serving job.

Each row's resource carries `run_id`, which Finelog promotes to the column of the
same name. It defaults to the Iris job id; `--run-id` sets an experiment identity.

Export and shutdown failures do not change training results or W&B ownership.
The Ray allowlist discards worker, address and task-name labels and never forwards
Ray's physical node or GPU families; Iris remains authoritative for host and GPU
telemetry, while centralized vLLM metrics stay with the serving job. Hardware
probes are not started. The frozen GPU runtime profile selects the telemetry
extra, and process shutdown gives Rigging at most two seconds to drain queued
records.

## Phase decomposition

Two span trees decompose the phases that were previously single wall timers. Both are off-by-default
or cheap-by-default, both publish through `phase_duration`, and both close against a **signed**
residual — a negative residual means a child is being counted inside another, and it is the only
automatic detector of that.

### `policy_train`, measured in the policy worker

Enabled by `trainer.policy_train_spans` (default off). `trainer.policy_train_spans_synchronize`
(default on) decides what the numbers mean, and it is not a free choice: CUDA kernels launch
asynchronously, so without a device synchronise a span measures kernel *launch* time and charges a
backward's real cost to whatever later call happens to block. With it, spans measure execution at
the cost of serialising the pipeline the run is otherwise trying to overlap.

**The two modes are not comparable, and the rows say so.** `clock_domain` is
`inclusive_wall`/`exclusive_wall` when synchronised and `inclusive_launch`/`exclusive_launch` when
not. Never mix them in one series.

`policy_ppo_train` and `policy_training_step` are *inclusive* of the spans beneath them; everything
else is exclusive. Rows carry `phase`, `root`, `parent`, `clock_domain`, `role=worker`, `rank`,
`step`.

### `generate`, measured on the driver's event loop

Enabled by `trainer.generate_spans` (default on: measured at ~21 ms against a ~98 s phase, and
there is no CUDA synchronise to pay for).

| span | contains | subtracted from the residual |
|---|---|---|
| `rollout_collect` | the fan-out over trajectories | yes |
| `rollout_assemble` | projecting results into a trainer batch | yes |
| `rollout_finalize` | the shared output epilogue | yes |
| `rollout_tokenize` | every tokenizer call — nested inside `rollout_collect` | **no** |
| `rollout_retain` | trajectory retention — nested inside `rollout_finalize` | **no** |
| `generate_span_residual` | `generate` minus the three above; signed | — |

### Counters, and why the waits are not spans

🚨 **A sum over concurrent coroutines is not a duration.** `generate` fans thousands of agent loops
over one event-loop thread, so the engine and environment waits sum to *far more* than the phase
that contains them — order 1e5 seconds against a ~98 s parent at large batch. They are published as
counters on their own instruments (`rollout_wait_seconds` in seconds, `rollout_count` unitless) with
attributes exactly `{counter, role=trainer, step}` — no rank, no phase, no parent, no
`clock_domain`. No counter name appears in the phase tree, so the phase publisher drops one even if
it is handed in.

| counter | meaning |
|---|---|
| `rollout_engine_await_seconds_sum` / `_count` | inference-engine wait, summed over concurrent trajectories |
| `rollout_env_await_seconds_sum` / `_count` | environment wait as the caller experiences it |
| `rollout_env_queue_seconds_sum` | submission until the executor picks the work up |
| `rollout_env_exec_seconds_sum` | the environment itself, stamped on the pool thread |
| `rollout_env_resume_seconds_sum` | the environment returning until the coroutine resumes — a direct read of **event-loop backlog** |
| `rollout_*_seconds_max` | the longest single trajectory's cumulative wait |

The three environment terms partition the caller-observed wait exactly. They are separate because
one bracket around the executor submission measures *queueing*, which on W pool threads serving N
trajectories grows as O(N²/W) — tens of seconds of "environment time" for an environment that did
not change, moving with the batch size.

`_seconds_max` exists because the phase is tail-latency-bound: the wall is set by the last
trajectory to finish, and a mean over thousands of them cannot separate a uniformly slow rollout
from a fast one with three stragglers.

### Support matrix — what each runner publishes

| runner / path | leaves + residual | waits | tails (`_seconds_max`) |
|---|---|---|---|
| `SkyRLGymTrajectoryRunner`, agent-loop | ✅ | ✅ | ✅ |
| `SkyRLGymTrajectoryRunner`, step-wise | ✅ | ✅ | ✅ |
| `SkyRLGymTrajectoryRunner`, **batched** | ✅ | ✅ | ❌ **absent** — one request serves a whole batch, so there is no per-trajectory tail to report |
| `HarborTrajectoryRunner`, `RolloutDispatcher` | ❌ **absent** | ❌ | ❌ |
| `MiniSweTrajectoryRunner` | ❌ **absent** | ❌ | ❌ |
| `FullyAsyncRayPPOTrainer` | ❌ **absent** | ❌ | ❌ — it keeps hundreds of overlapping `run()` calls in flight, and summing overlapping walls into one dict decomposes nothing |

**Absence is deliberate and is the honest signal.** A runner that has not bracketed its call sites
publishes nothing at all rather than a residual equal to its parent, because a full residual reads
as "generate is entirely unaccounted for" — a claim about the rollout when it is a fact about the
instrument. A runner opts in with `generate_spans_instrumented = True` after bracketing its own
waits; the flag is revoked automatically from a subclass that replaces `_run` without re-declaring
it.

### Volume

The policy tree publishes roughly 19 observations per rank per step, so about 1,200 at 64 ranks,
with `step` and `rank` as unbounded attributes and no sampling. The generate tree adds a handful of
driver rows per step. Budget accordingly before enabling the policy tree on a long run at scale.

### Known limits

- `n_tokens_dp_gt_*pct` are reduced with a **mean**, so they are a per-rank average rather than a
  global count. A sum would be worse: the reduction is over WORLD, and under sequence, context,
  expert or Megatron tensor/pipeline parallelism the replicas hold the same tokens, so a sum
  multiplies by the replication factor. A correct global count needs a data-parallel-group
  reduction that `Strategy.all_reduce` cannot currently express.
- `log_ratio_abs_p99` is a mean of per-rank p99 *approximations*, which is not a global quantile.
  Treat it as monitoring colour rather than a gate. `log_ratio_abs_max` **is** reduced with a max
  and is gate-grade.
- Allocator counters are scoped to one `ppo_train` call. Megatron overrides `ppo_train` and does not
  publish them.
