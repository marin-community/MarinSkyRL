# Async RL telemetry: what we record and why

This note explains the telemetry MarinSkyRL exports for reinforcement-learning runs, starting from the
questions a person watching a run asks. Each section names the question, the metrics that answer it,
the code that records them, and the Grafana panels that show them. The panels are on
**RL Post-training (async)** (`marin-async-rl`) and **RL Post-training (sync)** (`marin-rl-runs`);
`docs/grafana-rl-runs.md` explains how a run gets onto them, and `skyrl-train/docs/telemetry.md` explains
the export path.

Every record carries the run id, the execution uid, the process role and `training_type` (`sync` or
`async`), so each dashboard lists only the runs of its own training type. Composing the launch document
sets `runtime.training_type` from the resolved entrypoint and `trainer.placement.colocate_all`:
`fully_async` is async, `terminal_bench` is async when `colocate_all` is false, the generate entrypoints
have no training type, and every other entrypoint is sync. The Iris task runtime exports it as
`SKYRL_TRAINING_TYPE` before Ray starts, so every Ray actor inherits it. Panel numbers below are the
panel ids on the async dashboard unless marked "sync".

## 1. Is the run alive and reporting?

A run can hang, lose a process, or stop exporting while its job still shows as running. These records
are on whenever the run has a telemetry endpoint.

| Question | Metric | Recorded by | Panel |
|---|---|---|---|
| Did every process start, and did any exit? | `lifecycle` and `terminal` events | every trainer, worker and controller process (`telemetry.ProcessTelemetry`) | 4 Process lifecycle |
| Is the trainer taking optimizer steps? | `policy_step` gauge | trainer driver | 2 Optimizer and synced policy steps |
| Are new weights reaching vLLM? | `weight_sync_completed` event with the installed version | trainer driver | 2 (synced series) |
| Is telemetry itself being dropped? | `telemetry_lost_records`, `telemetry_rejected_records` | the exporter in each process | 28 Exporter and nonfinite observations |
| Did a metric go NaN or infinite? | `training_nonfinite_values` | trainer driver | 28 |

## 2. Is the loop keeping the trainer fed?

The fully async loop is healthy when generation produces one update's groups in less time than the
trainer takes to consume them, and when the groups it trains on are fresh.

| Question | Metric | Recorded by | Panel |
|---|---|---|---|
| How many tokens are generated and trained per second? | `work_completed` by kind (generated, trained response, trained loss tokens) | generation workers, trainer driver | 3 Response tokens generated and trained on / s |
| Is the trainer waiting for data? | `phase_duration_seconds` for `wait_for_generation_buffer`; buffer-wait fraction of the core wall | trainer driver | 6 Driver walls, 36 Core wall fractions |
| How full is the buffer of finished groups? | `rollout_queue_depth`, `rollout_capacity` | trainer driver | 8 Completed buffer depth and capacity |
| How long does a finished group wait before training? | `rollout_buffer_dwell_seconds` | trainer driver | 10 Buffer dwell of trained groups |
| How stale is the data the trainer uses? | `rollout_staleness_steps`; `consumed_staleness` events with tokens per group | trainer driver | 11 Policy staleness at training, 51 and 52 Staleness of trained groups |
| What happens to every generated group? | `rollout_groups`, `rollout_group_tokens` by disposition (trained, rejected as stale, dropped at shutdown) | trainer driver | 19, 20 |
| Does generation overlap training? | `rollout_call` finish times joined to `async_phase_window` training intervals | trainer driver | 18 Rollouts completing during policy training |

## 3. Where does each component spend its time and memory?

Once the loop is known to be starved or backed up, these records show which component is responsible.

**Generation.** `observe_rollout_call` times each trajectory-runner call and splits it into collect,
assemble, finalize, tokenize and retain phases; `rollout_wait` records the waits inside it (prompt,
worker slot, enqueue, model call, environment queue, execution and resume). The driver samples its own
event-loop lag. Panels 7, 12, 15, 16. vLLM's own token counters, joined to the trainer's phase windows,
give vLLM's service rate during each trainer phase (panel 42). The synchronous trainer records the same
rollout-call breakdown under `generate_spans`.

**Trainer.** `Timer` spans give the driver's step, buffer wait, training and weight-sync walls
(panel 6). `async_step_metrics` derives the core and cycle walls, useful tokens per second and per
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
| Mean and quantiles of \|delta\| | mean, p50, p95, p99, p99.9 and max of `abs(delta)` | How far apart the two models are on typical and worst tokens. A small mean with a large p99.9 means a few tokens disagree strongly while the rest agree. |
| Mean of delta | `mean(delta)` | The direction of the disagreement: negative when the trainer gives vLLM's samples lower probability than vLLM did. |
| PPO-window clip pressure | fraction of tokens with `w` below `1 - eps_low` (lower) or above `1 + eps_high` (upper) | The share of the batch that PPO's clip already treats as off-policy before any update, from mismatch alone. Those tokens contribute a clipped gradient in the first update step. |
| Fraction outside [0.5x, 2x] | fraction of tokens with `abs(delta) > log 2` | Tokens whose probability halved or doubled between sampling and training. |
| Fraction below 1e-5 | fraction of tokens with `w < 1e-5` | Tokens the trainer considers almost impossible although vLLM sampled them, for example from a sampling setting the trainer does not model or a numerical failure. |
| ESS fraction | `(sum w)^2 / (N * sum w^2)` | Kish's effective sample size of the importance weights as a fraction of the token count. 1 when the models agree; small when a few tokens would dominate an importance-weighted gradient. |
| k1 and k3 KL | `mean(-delta)` and `mean(exp(delta) - delta - 1)` | Two estimators of KL(vLLM ‖ trainer) from Schulman's "Approximating KL Divergence". k1 is unbiased but can be negative on a batch; k3 is never negative and has lower variance, and is the form GRPO uses for its KL term. |
| chi-squared | `mean(w^2) - 1` | The variance of the importance weights; ESS is roughly `N / (1 + chi-squared)`, so a growing chi-squared is the same event as a shrinking ESS. |

Split by staleness, these separate two causes. At staleness 0 the trainer and vLLM hold the same weights,
so any mismatch comes from numerics: kernels, precision and sampling processors (panel 54). Growth across
staleness buckets is the policy drift that stale data adds (panel 55). Panels 30 to 32 and 44 show the
pooled values, 57 the dependence on token position, and 47 and 48 batches of uniform staleness. Panel 56
shows the same statistics within one update, the trainer against its own pre-update log probabilities.
Panels 46 and 59 show how often TIS and the off-policy masks act on these tokens.

**Gradient direction.** `GradientDirectionTracker` records the cosine between successive policy-update
gradients, with the minimum and maximum across the optimizer steps of one training call and the gradient
norm (panel 58).
Xu et al., "GAC: Stabilizing Asynchronous RL Training for LLMs via Gradient Alignment Control"
(arXiv:2603.01501) report persistently high cosine between consecutive policy gradients under
asynchronous training, near-orthogonal updates under synchronous training, and tie the high-cosine regime
to divergence. The cosine lets us check whether our staleness settings enter that regime.

On fresh data, consecutive gradients come out near-orthogonal for two reasons. Each step removes most of
the objective's slope along the direction it just moved, as an exact line search would. The minibatch
noise that remains is high-dimensional, and independent high-dimensional vectors are nearly orthogonal.
Stale data breaks the first reason. A batch sampled several versions ago carries the reward signal of the
policy that sampled it, so it keeps pointing the trainer in the direction the last few updates already
moved. The trainer sees the effect of an update only after the staleness delay, which acts like momentum
nobody configured: successive updates stay aligned, and their sum can overshoot. A cosine that stays near
1 across updates is that feedback delay made visible.

## 5. Gates, defaults and costs

Records reach Finelog only when the run has a telemetry endpoint, run id and execution uid, which the
Iris task runtime sets. Within that, each family has its own switch.

| Family | Gate | Default | Cost when on | Read by |
|---|---|---|---|---|
| Liveness, steps, buffer, staleness, phase walls | none | on | a few records per step | 2, 3, 4, 6, 8, 11, 14, 28; sync board |
| Trainer scalars and loop events | `trainer.training_metrics` | on | one record per scalar per optimizer step | most of sections 1, 2 and 4 |
| Async rollout spans and waits | `trainer.async_spans` | on | several records per rollout call; the largest record volume | 7, 10, 12, 15, 16, 18 to 20, 42, 53 |
| Sync rollout spans | `trainer.generate_spans` | on | several records per rollout call | sync board drill-downs |
| Learner memory and Megatron phase walls | `trainer.policy_train_spans` | on | a few records per phase per rank; no CUDA synchronization | 16, 26, 27, 41 |
| Pooled ratio diagnostics | `trainer.algorithm.ratio_diagnostics.pooled` | null: on for Megatron, off elsewhere | gathers each rank's statistics on every optimizer minibatch | 30 to 32, 44, 54, 55, 57 |
| Gradient direction | `trainer.algorithm.grad_cosine.enabled` | off; supported on fsdp, fsdp2 and megatron | an fp32 copy of every gradient shard and one all-reduce per update | 58 |

The light families are on by default because a run without them cannot be diagnosed from the
dashboards, and they add records without adding GPU work. The gradient cosine stays off because it holds
a second copy of the gradients; turn it on for a run that is being studied.

Two families measure only on some trainer strategies. Config validation resolves both against
`trainer.strategy` with one table: a null switch turns on where the strategy supports the family and off
elsewhere, with one info log naming what was left off, and an explicit true on an unsupported strategy
fails validation.
