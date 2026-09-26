# Async RL telemetry

MarinSkyRL exports run progress, throughput, timing, memory and off-policy metrics to Finelog. The
**RL Post-training (async)** (`marin-async-rl`) and **RL Post-training (sync)** (`marin-rl-runs`)
dashboards display them. [Watching an RL run on Grafana](../grafana-rl-runs.md) covers run selection
and [Telemetry](../../skyrl-train/docs/telemetry.md) covers the export path.

Every record carries the run id, the execution uid, the process role and `training_type`. Launch
composition sets `training_type` from `trainer.rollout_buffer.max_staleness_steps`: `sync` at 0 and
`async` above it; the generate entrypoints have none. Both modes run the same training loop and record
the same families. Panel numbers refer to the async dashboard.

## Liveness and flow

| Question | Records | Panels |
|---|---|---|
| Did every process start or exit? | `lifecycle`, `terminal` | 4 |
| Is the trainer stepping, and do new weights reach vLLM? | `policy_step`, `weight_sync_completed` | 2 |
| Is telemetry dropped, or did a metric go non-finite? | `telemetry_lost_records`, `telemetry_rejected_records`, `training_nonfinite_values` | 28 |
| How many tokens are generated and trained per second? | `work_completed` by `work_kind` | 3 |
| Is the trainer waiting for data? | `phase_duration_seconds` for `wait_for_generation_buffer`; `async/performance/buffer_wait_fraction` | 6, 36 |
| How full is the buffer, and how long do groups wait in it? | `rollout_queue_depth`, `rollout_capacity`, `rollout_buffer_dwell_seconds` | 8, 10 |
| How stale is the trained data? | `rollout_staleness_steps`, `consumed_staleness` | 11, 51, 52 |
| What happens to each generated group? | `rollout_groups`, `rollout_group_tokens` by `disposition` | 19, 20 |
| Does generation overlap training? | `rollout_call` windows joined to `async_phase_window` | 18 |
| How long does each weight sync take? | `phase_duration_seconds` for `sync_weights` and its stages | 14, 53 |
| Is the model learning? | reward, evaluation, stop-reason and optimizer scalars in `training_metric_value` | 22 to 24, 40, 49, 50 |

The rollout buffer judges each committed group. Its disposition is `consumed` when a batch takes it,
an admission rejection such as `stale`, `fully_masked` or `duplicate_uid`, or
`insufficient_reward_spread` from dynamic sampling. Buffer dwell runs from commit to that judgment; a
group restored from a checkpoint has no dwell, and a group still buffered at shutdown has no
disposition.

## Time and memory

Each rollout call records its wall time split into collect, assemble and finalize, with tokenize under
collect, retain under finalize and the remainder as `rollout_call_residual`. Its waits cover the model
call, the environment's queue, execution and resume, and the enqueue to the buffer; the dispatch loop
also records its waits for a prompt and a lease slot (panels 7, 12, 15, 16). A rollout worker in another
process reports only the call's wall and response tokens. The driver samples its event-loop lag, and vLLM's
token counters joined to the trainer's phase windows give vLLM's rate during each phase (panel 42).

The driver's step, buffer wait, training and weight-sync walls feed panel 6, and
`async/performance/*` gives the core and cycle walls, their fractions and useful tokens per second and
per configured GPU (panels 34, 35, 38, 39, 43). Each policy update splits into the forward-backward
schedule, pipeline metric broadcast, optimizer step, world reduction, final barrier and
`ppo_train_residual` (panels 16, 26, 27). `cuda_memory_observation` records allocator peaks, reserved and
allocated bytes and free device memory around each policy phase per rank (panel 41).

## Off-policy health

For every trained token the driver computes `delta = log p_trainer(token) - log p_vLLM(token)` and
`w = exp(delta)`, the importance weight from vLLM's distribution to the trainer's. Identical models give
`delta = 0` and `w = 1`.

| Statistic | Definition | Reading |
|---|---|---|
| \|delta\| mean and quantiles | mean, p95, p99, p99.9 and max of `abs(delta)` | typical and worst disagreement |
| Mean of delta | `mean(delta)` | negative when the trainer assigns vLLM's samples less probability |
| Clip pressure | fraction with `w < 1 - eps_low` or `w > 1 + eps_high` | tokens PPO clips from mismatch alone |
| Fraction outside [0.5x, 2x] | fraction with `abs(delta) > log 2` | tokens whose probability halved or doubled |
| Fraction below 1e-5 | fraction with `w < 1e-5` | sampled tokens the trainer considers nearly impossible |
| ESS fraction | `(sum w)^2 / (N * sum w^2)` | 1 when the models agree; small when a few tokens dominate |
| k1 and k3 KL | `mean(-delta)` and `mean(exp(delta) - delta - 1)` | estimates of KL(vLLM ‖ trainer); k3 is never negative |
| chi-squared | `mean(w^2) - 1` | importance-weight variance; ESS is about `N / (1 + chi-squared)` |

The `pooled` bucket covers every loss token and `staleness0` the tokens sampled with the trainer's
current weights; both report every statistic. The other staleness buckets and the first 256, middle and
last 256 response positions report token counts, finite fraction, mean and mean absolute delta, and the
fraction outside [0.5x, 2x]. Staleness 0 isolates numerical mismatch (panel 54); growth across buckets
is policy drift (panel 55). Panels 30 to 32 and 44 show the pooled values, 57 the position dependence and
47 and 48 uniform-staleness batches. Panel 56 shows the trainer against its own pre-update log
probabilities, pooled across data-parallel ranks.

## Switches and costs

Records reach Finelog only when the Iris task runtime sets the telemetry endpoint, run id and execution
uid. Each family below has its own switch, on by default.

| Family | Switch | Cost | Panels |
|---|---|---|---|
| Liveness, steps, buffer, staleness, phase walls | none | a few records per step | 2, 3, 4, 6, 8, 11, 14, 28 |
| Trainer scalars, loop events and mismatch statistics | `trainer.training_metrics` | one record per scalar per step; mismatch statistics take about 0.2 s of driver CPU at 1.3M loss tokens | sections above, 30 to 32, 44, 47, 48, 54 to 57 |
| Rollout calls, waits, dispositions and loop windows | `trainer.rollout_spans` | several records per rollout call | 7, 10, 12, 15, 16, 18 to 20, 42, 53 |
| Policy-update phases and learner memory | `trainer.policy_train_spans` | a few records per phase per rank | 16, 26, 27, 41 |
