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

Enabled by `trainer.policy_train_spans` (default off). **FSDP2 only** — the Megatron worker
overrides `ppo_train` and does not bracket Megatron Core's pipeline scheduler, so the flag would
publish no spans at all there; `validate_cfg` rejects the combination rather than letting a run
measure nothing. `trainer.policy_train_spans_synchronize`
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

Enabled by `trainer.generate_spans`, default on. The tree adds no CUDA synchronise, and a
microbenchmark of the context managers alone costs ~21 ms against a ~98 s phase. That is a bound on
the bookkeeping, **not a measured end-to-end overhead**: no matched spans-on/spans-off pair has been
run at the same SHA, and a single run cannot measure its own overhead.

🚨 **The ~21 ms does not include publishing.** `publish_driver_counters` runs one blocking
`telemetry.flush(1.0)` in the trainer's step epilogue, so a degraded endpoint costs up to **1 s per
step** — about fifty times the bookkeeping bound, and the term that will dominate any A/B you run.
The worker path pays the same 1 s cap on its own return.

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
| `rollout_trajectory_count` | how many trajectory scopes closed — **the denominator the tail must be read against** |

⚠️ **`rollout_engine_await` is not purely engine time on the `prompts=` call form.** Callers that pass
text rather than token ids -- `collect_batched`, and `agent_loop` under `retokenize_chat_history` --
await the whole `InferenceEngineClient.generate` inside the wait, and that call templates on the
driver's event-loop thread before any engine is touched. That templating is now bracketed as
`rollout_tokenize`, so it is **visible in the tree**, but it is **still inside the wait counter** --
the bracket cannot narrow a region several frames up the stack. So on those paths the two overlap:
subtract `rollout_tokenize` from `rollout_engine_await` before reading the latter as engine time, and
do not sum them.

The complete repair is to hoist the templating to the caller and pass `prompt_token_ids=`, which is a
**behaviour** change rather than an instrumentation one -- `chat_template_kwargs` is documented as
incompatible with `batched=True` precisely because the engine client owns templating there. It is
deliberately left as follow-up work rather than smuggled into a telemetry change.

⚠️ **`_count` and `_seconds_max` are different populations, and mixing them is the easy mistake.**
`_count` counts timed *calls*, of which one trajectory makes several, so `sum / _count` is a mean
**per call**. `_seconds_max` is a max **per trajectory**. Use `sum / rollout_trajectory_count` for a
per-trajectory mean, which is the number the tail is comparable to. A trajectory that closed without
ever waiting contributes a real `0.0` to the max and a `1` to **`rollout_trajectory_count`** — not to
the await `_count`s, which only a timed call increments. So the *trajectory* population is complete
while the await counts are over calls that actually waited.

🚨 **`rollout_trajectory_count` is ABSENT on the batched path**, where no trajectory scope closes. It
is a denominator, so a seeded `0.0` there would make every per-trajectory mean a division by zero;
absent, you can tell the mean is not derivable. Check for the row before dividing.

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
| `SkyRLGymTrajectoryRunner`, **batched** | ✅ | ✅ | ❌ **absent** — one request serves a whole batch, so there is no per-trajectory tail to report, and `rollout_trajectory_count` is absent with them |
| `HarborTrajectoryRunner`, `RolloutDispatcher` | ❌ **absent** | ❌ | ❌ |
| `MiniSweTrajectoryRunner` | ❌ **absent** | ❌ | ❌ |
| `FullyAsyncRayPPOTrainer` | ❌ **absent** | ❌ | ❌ — it keeps hundreds of overlapping `run()` calls in flight, and summing overlapping walls into one dict decomposes nothing |

**Absence is deliberate and is the honest signal.** A runner that has not bracketed its call sites
publishes nothing at all rather than a residual equal to its parent, because a full residual reads
as "generate is entirely unaccounted for" — a claim about the rollout when it is a fact about the
instrument.

**Opting in takes TWO certificates, and a runner needs both.** They cover different halves and
neither implies the other:

1. **The runner class** declares `generate_spans_instrumented = True` after bracketing its own
   waits. It is revoked automatically from a subclass that overrides any of
   `TrajectoryRunner.BRACKETED_METHODS` — `_run`, `agent_loop`, `collect_batched` — without
   re-declaring it, because those are the bracketed loops.
2. **The collector class** declares the same flag, and the runner reads the collector's own
   `__dict__` so a subclass cannot inherit it. This half matters because the collector is
   **injected**: `pipeline=...` is the supported extension point, and a caller-supplied collector
   that brackets nothing is not covered by anything the runner class says about itself.

⚠️ **If you inject a collector, declare the flag on it.** Following step 1 alone and supplying your
own collector publishes nothing, silently — the runner is certified and the collector is not, so the
conjunction is false. That is the honest outcome, but it is not obvious from step 1.

### Volume

The policy tree publishes roughly 19 observations per rank per step, so about 1,200 at 64 ranks,
with `step` and `rank` as unbounded attributes and no sampling. The generate tree adds a handful of
driver rows per step. Budget accordingly before enabling the policy tree on a long run at scale.

### Known limits

- `rank_tokens_real` / `rank_tokens_padded` are **this rank's** tokens, published per rank and never
  reduced. Summing them across rows gives the global total only when every rank holds distinct data:
  under sequence, context, expert or Megatron tensor/pipeline parallelism the replicas hold the same
  tokens and the sum is multiplied by the replication factor. The `rank_` prefix is there to make
  that visible at the point of use.
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
- **The worker subtree hangs off `train_critic_and_policy`, not off the driver phase that ran.**
  With a critic configured and `colocate_all` off the trainer runs `ppo_train` under
  `Timer("policy_critic_overlap_train")` instead of `Timer("policy_train")` (`trainer.py:1964` vs
  `:1977`), so neither of those is a safe parent -- naming one would orphan the whole subtree on the
  other path. `train_critic_and_policy` (`trainer.py:731`) wraps both branches and is always
  published, so worker rows name it and the two driver phases are their siblings rather than their
  ancestors.

  ⚠️ **The cost is real, and it is not merely "one level of resolution."** The driver's `policy_train`
  and the worker's `policy_ppo_train` are now siblings under one parent **while the first contains
  the second**, and nothing in the data expresses that containment any more -- it used to be the one
  relation the tree stated. **So do not subtract one from the other, and do not sum them.** The
  orphaning this avoids needs `critic_model is not None` **and** `colocate_all` false, and
  `colocate_all` defaults **true** (`ppo_base_config.yaml:55`) -- so the trade buys a non-default
  corner at a cost paid on the default path. It is taken because the orphaning is silent and total
  while this is merely coarse.
