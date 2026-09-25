# Async RL telemetry

MarinSkyRL exports run progress, throughput, timing, memory and off-policy metrics to Finelog.
**RL Post-training (async)** (`marin-async-rl`) and **RL Post-training (sync)** (`marin-rl-runs`)
display them. See [Watching an RL run on Grafana](../grafana-rl-runs.md) for run selection and
[Telemetry](../../skyrl-train/docs/telemetry.md) for the export path.

Every record carries the run id, the execution uid, the process role and `training_type` (`sync` or
`async`), so each dashboard lists only the runs of its own training type. Composing the launch document
sets `runtime.training_type` from the resolved entrypoint and `trainer.placement.colocate_all`:
`fully_async` is async, `terminal_bench` is async when `colocate_all` is false, the generate entrypoints
have no training type, and every other entrypoint is sync. The Iris task runtime exports it as
`SKYRL_TRAINING_TYPE` before Ray starts, so every Ray actor inherits it. Panel numbers below are the
panel ids on the async dashboard unless marked "sync".

## 1. Is the run alive and reporting?

| Question | Metric | Recorded by | Panel |
|---|---|---|---|
| Did every process start, and did any exit? | `lifecycle` and `terminal` events | every process under its role: `driver`, `trainer`, `worker`, and `controller` for the Ray-metrics forwarder on the Iris head node (`telemetry.ProcessTelemetry`) | 4 Process lifecycle |
| Is the trainer taking optimizer steps? | `policy_step` gauge | trainer driver | 2 Optimizer and synced policy steps |
| Are new weights reaching vLLM? | `weight_sync_completed` event with the installed version; needs `trainer.training_metrics` | trainer driver | 2 (synced series) |
| Is telemetry itself being dropped? | `telemetry_lost_records`, `telemetry_rejected_records` | the exporter in each process | 28 Exporter and nonfinite observations |
| Did a metric go NaN or infinite? | `training_nonfinite_values`; needs `trainer.training_metrics` | trainer driver | 28 |

## 2. Is the loop keeping the trainer fed?

| Question | Metric | Recorded by | Panel |
|---|---|---|---|
| How many tokens are generated and trained per second? | `work_completed` by kind (generated, trained response, trained loss tokens) | generation workers, trainer driver | 3 Response tokens generated and trained on / s |
| Is the trainer waiting for data? | `phase_duration_seconds` for `wait_for_generation_buffer`; buffer-wait fraction of the core wall | trainer driver | 6 Driver walls, 36 Core wall fractions |
| How full is the buffer of finished groups? | `rollout_queue_depth`, `rollout_capacity` | trainer driver | 8 Completed buffer depth and capacity |
| How long does a finished group wait before training? | `rollout_buffer_dwell_seconds` | trainer driver | 10 Buffer dwell of trained groups |
| How stale is the data the trainer uses? | `rollout_staleness_steps`; `consumed_staleness` events with tokens per group | trainer driver | 11 Policy staleness at training, 51 and 52 Staleness of trained groups |
| What happens to every generated group? | `rollout_groups`, `rollout_group_tokens` by disposition: `consumed`; `stale_enqueue` when the producer finds its group stale; the admission rejection (`stale`, `fully_masked`, `duplicate_uid` and the other `AdmissionRejection` values); `insufficient_reward_spread` from dynamic sampling; `epoch_end_drain`. A group still buffered at shutdown gets no disposition. | trainer driver | 19, 20 |
| Does generation overlap training? | `rollout_call` finish times joined to `async_phase_window` training intervals | trainer driver | 18 Rollouts completing during policy training |

## 3. Where does each component spend its time and memory?

**Generation.** `observe_rollout_call` times each trajectory-runner call and splits it into collect,
assemble, finalize, tokenize and retain phases; `rollout_wait` records the waits inside it (model call,
environment queue, execution and resume). `async_wait` records the producer's waits around the call: for
a prompt, for a worker slot, and to enqueue the finished group. The driver samples its own event-loop
lag. Panels 7, 12, 15, 16. vLLM's own token counters, joined to the trainer's phase windows, give vLLM's
service rate during each trainer phase (panel 42). The synchronous trainer records the same rollout-call
breakdown under `generate_spans`.

**Trainer.** `Timer` spans give the driver's step, buffer wait, training and weight-sync walls
(panel 6). The fully async trainer times batch preparation as
`assemble_generation_group_mini_batch`, `postprocess_trajectory_batch` and `convert_to_training_input`,
all under `step`. `async_step_metrics` derives the core and cycle walls, useful tokens per second and per
configured GPU, and cumulative core GPU-hours (panels 34, 35, 38, 39, 43). On Megatron,
`MegatronTrainTimings` splits one policy update into the forward-backward schedule, pipeline metric
broadcast, optimizer step, world reduction and final barrier, with the unattributed remainder as a
residual (panels 26, 27, 16).

**Weight sync.** Each sync's wall and its pause-to-resume stages (panels 14, 53).

**Memory.** `LearnerMemory` records allocator peaks, reserved and allocated bytes and free device
memory around each policy phase, per rank (panel 41).

## 4. Is the model learning, and is off-policy data hurting it?

**Learning.** Reward and the fraction of groups with any reward spread (panel 22), evaluation scores,
response lengths and stop reasons (panels 23, 40, 49, 50), and the optimizer's entropy, gradient norm,
KL and loss (panel 24).

**Off-policy health.** The trainer learns from tokens vLLM sampled. For every trained token the trainer
computes `delta = log p_trainer(token) - log p_vLLM(token)`, and `ratio_statistics` summarizes the
distribution of `delta`:

`w = exp(delta)` is the importance weight that would reweight a token sampled by vLLM to the trainer's
distribution. When the two agree, every `delta` is 0 and every `w` is 1.

| Statistic | Definition | How to read it |
|---|---|---|
| Mean and quantiles of \|delta\| | mean, p95, p99, p99.9 and max of `abs(delta)` | How far apart the two models are on typical and worst tokens. A small mean with a large p99.9 means a few tokens disagree strongly while the rest agree. |
| Mean of delta | `mean(delta)` | The direction of the disagreement: negative when the trainer gives vLLM's samples lower probability than vLLM did. |
| PPO-window clip pressure | fraction of tokens with `w` below `1 - eps_low` (lower) or above `1 + eps_high` (upper) | The share of the batch that PPO's clip already treats as off-policy before any update, from mismatch alone. Those tokens contribute a clipped gradient in the first update step. |
| Fraction outside [0.5x, 2x] | fraction of tokens with `abs(delta) > log 2` | Tokens whose probability halved or doubled between sampling and training. |
| Fraction below 1e-5 | fraction of tokens with `w < 1e-5` | Tokens the trainer considers almost impossible although vLLM sampled them, for example from a sampling setting the trainer does not model or a numerical failure. |
| ESS fraction | `(sum w)^2 / (N * sum w^2)` | Kish's effective sample size of the importance weights as a fraction of the token count. 1 when the models agree; small when a few tokens would dominate an importance-weighted gradient. |
| k1 and k3 KL | `mean(-delta)` and `mean(exp(delta) - delta - 1)` | Two estimators of KL(vLLM ‖ trainer) from Schulman's "Approximating KL Divergence". k1 is unbiased but can be negative on a batch; k3 is never negative and has lower variance, and is the form GRPO uses for its KL term. |
| chi-squared | `mean(w^2) - 1` | The variance of the importance weights; ESS is roughly `N / (1 + chi-squared)`, so a growing chi-squared is the same event as a shrinking ESS. |

The driver computes every statistic above for all loss tokens (the `pooled` bucket) and for staleness-0
tokens. The other staleness buckets, and every bucket's first, middle and last token positions, carry the
token counts, finite fraction, mean and mean absolute delta, and the fraction outside [0.5x, 2x].

Staleness buckets separate two causes. At staleness 0 the trainer and vLLM hold the same weights,
so any mismatch comes from numerics: kernels, precision and sampling processors (panel 54). Growth across
staleness buckets is the policy drift that stale data adds (panel 55). Panels 30 to 32 and 44 show the
pooled values, 57 the dependence on token position, and 47 and 48 batches of uniform staleness. Panel 56
shows the same statistics within one update, the trainer against its own pre-update log probabilities.
Panels 46 and 59 show how often TIS and the off-policy masks act on these tokens.

## 5. Gates, defaults and costs

Records reach Finelog only when the run has a telemetry endpoint, run id and execution uid, which the
Iris task runtime sets. Within that, each family has its own switch.

| Family | Gate | Default | Cost when on | Read by |
|---|---|---|---|---|
| Liveness, steps, buffer, staleness, phase walls | none | on | a few records per step | 2, 3, 4, 6, 8, 11, 14, 28; sync board |
| Trainer scalars and loop events | `trainer.training_metrics` | on | one record per scalar per optimizer step | most of sections 1, 2 and 4 |
| Trainer/vLLM mismatch statistics (`policy/mismatch/{pooled,staleness*}/*`) | `trainer.training_metrics`, when rollout logprobs exist | on | CPU work on the driver in each step's critical path, linear in loss tokens: one pass for the bucket moments and, for the pooled and staleness-0 buckets, one selection plus a sort of the top 5% of \|delta\|. About 0.2 s at 1.3M loss tokens and 1.5 s at 5.4M on one CPU thread | 30 to 32, 44, 47, 48, 54, 55, 57 |
| Async rollout spans and waits | `trainer.async_spans` | on | several records per rollout call; the largest record volume | 7, 10, 12, 15, 16, 18 to 20, 42, 53 |
| Sync rollout spans | `trainer.generate_spans` | on | several records per rollout call | sync board drill-downs |
| Learner memory and Megatron phase walls | `trainer.policy_train_spans` | on | a few records per phase per rank; no CUDA synchronization | 16, 26, 27, 41 |
| Pooled within-update log-ratio statistics (`policy/log_ratio_*`) | `trainer.algorithm.ratio_diagnostics.pooled` | null: on for Megatron, off elsewhere | gathers each rank's statistics on every optimizer minibatch; off, each rank summarizes its own tokens | 56, 57 (`policy/log_ratio_pos_*`) |

The light families are on by default because a run without them cannot be diagnosed from the
dashboards, and they add records without adding GPU work.

Pooled log-ratio statistics measure only on Megatron. Config validation resolves the switch against
`trainer.strategy`: null turns on where the strategy supports pooling and off elsewhere, with one info log
naming what was left off, and an explicit true on an unsupported strategy fails validation.
