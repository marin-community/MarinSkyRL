# Iris RL diagnostic signals

Use these semantics with evidence captured through `coreweave.md`.

## Diagnostic mode

Determine the mode from the launched configuration before choosing probes.

- **Agentic:** Harbor, Daytona, terminal-bench, or another sandboxed agent harness. These runs can
  produce per-trial results, exceptions, transcripts, and timing artifacts under `trace_jobs/`.
- **Standard:** dataset-backed rewards without an agent harness. These runs have no per-trial
  sandbox artifacts; absence of `trace_jobs/` is expected and is not an evidence error.

Parse synchronized metrics with `infra/rl_cleanup/parse_skyrl_metrics.py`; it detects the training-log
serialization independently of the diagnostic mode. Pass `--trace_jobs_dir` when agentic traces are
not adjacent to the logs. Treat empty parser output as a failed probe and inspect the log serialization.

## Progress

Trainer phase boundaries and the training-step counter are the progress truth. Fresh inference
heartbeats, trace counts, and controller state can coexist with a stalled policy mesh. Compare log
freshness with the run's observed cadence; long agent episodes and large forwards can be quiet.

For colocated jobs, inspect policy and inference meshes separately. Read both aggregate and
per-actor logs for deaths, collectives, and rank reachability. Most diagnostics are rank-zero only,
so absence on other ranks is not proof they skipped the operation.

## Engine saturation

Interpret the tuple of scheduler waiting depth, running requests, KV-cache occupancy, power, and
memory-bandwidth activity.

- Persistent waiting plus active KV use indicates decode saturation.
- Empty waiting queues, low KV use, and low power indicate under-feeding.
- Sawtooth running counts during policy training can be healthy backpressure when generation-buffer
  wait time is negligible.

For agentic runs, demand and supply are coupled:

```text
demand = n_concurrent_trials
supply = max_num_seqs * num_inference_engines
```

For standard runs, use the rollout batch as demand:

```text
demand = train_batch_size * n_samples_per_prompt
supply = max_num_seqs * num_inference_engines
```

Before attributing starvation, measure coordinator CPU/GIL pressure, staleness throttling, and
whether engine requests arrive and drain quickly.

## Agentic rollout quality

Inspect actual trial outputs, rewards, verifier results, stop reasons, and exceptions.

- All-zero rewards from the first step usually indicate a data, sandbox, verifier, or weight-sync
  failure before a learning conclusion.
- Very short or empty attempts suggest a dead request path or broken agent loop.
- Incoherent output immediately after weight synchronization points toward reload ordering or stale
  weights; correlate with reload logs and TIS metrics.
- Sandbox-not-found errors during queued work point toward idle-reap policy.
- Uniform environment-setup failures point toward image/dependency or provider configuration; name
  the concrete exception before attributing cause.

## Standard rollout quality

Standard runs have no Harbor result files, sandbox transcripts, or per-trial exception artifacts.
Use these substitutes for the rollout gate:

- `reward/avg_raw_reward` and the trainer reward histogram for reward level and distribution;
- verifier or environment exceptions surfaced in trainer logs;
- banked-step cadence and `checkpoints/global_step_N` freshness for durable progress evidence;
- `generate/avg_num_tokens` for empty-output or generation-collapse detection;
- engine `Running` and `Waiting` counts relative to `train_batch_size * n_samples_per_prompt` for
  request flow and saturation.

Missing `trace_jobs/` is not an `ERROR` in standard mode. A rollout `ERROR` requires unavailable or
contradictory standard evidence, such as missing trainer metrics and logs, not absent agentic
artifacts.

## Training dynamics

Use at least 10 completed training steps unless the campaign requires a larger window. Treat a
shorter learning-quality window as insufficient evidence and report the actual series:

- reward and completion quality;
- entropy;
- TIS importance-ratio, capped/skipped, and exact-match metrics;
- policy loss and gradient norms;
- per-phase durations, especially generation-buffer wait.

Read a phase share next to the step index it came from: the first step of a fully asynchronous run
pays a one-time rollout-buffer fill, so an average over few steps reports that startup cost as
steady state.

Metric sinks may add prefixes. Match the semantic suffix rather than assuming one serialized key.
Trainer metrics can lag live rollouts under asynchronous generation. For agentic runs, require
trial-level agreement before a learning-quality kill recommendation. For standard runs, reconcile
the trainer metric stream with banked-step/checkpoint cadence and verifier errors in trainer logs.

A plateau alone is not collapse. For a degradation verdict, identify the mechanism, confirm it over
the declared window, show that it is not recovering, and distinguish model behavior from optimizer
signal. Extreme ratio, loss, or gradient changes following synchronization indicate stale or wrong
weights more strongly than reward alone.

## Agentic per-trial duty cycle

Trial timing commonly separates environment setup, agent setup, agent execution, and verifier.
Per-request API timings isolate generation within agent execution; the remainder approximates tool
work. Report bounded-sample medians and tails with the sampling window. Startup provisioning bursts
are not steady-state sandbox churn.

Per-trial duty cycle is not applicable to standard runs. Report it as `N/A`. Include standard
`timing/*` phases such as `wait_for_generation_buffer`, `policy_train`, and `sync_weights` in the
dynamics phase table instead.

## Configuration and resume failures

- Hydra struct errors occur before distributed training when a launched override is absent from the
  baked base configuration.
- Resolve checkpoint directories from run metadata; a top-level artifact directory alone proves
  nothing.
- Chained resume can reach or exceed the configured ceiling. Treat an already completed step budget
  as terminal even when a redundant tail attempt errors.

## Restart burn

Inspect every attempt's first causal error. The same deterministic failure on repeated attempts is a
kill-and-fix recommendation; a recovered transient is not. Get counts from controller state and
errors from the captured logs.
