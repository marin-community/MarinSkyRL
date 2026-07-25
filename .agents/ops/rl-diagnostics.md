# Reading a running RL job — log vocabulary, health signals, and known failure signatures

How to interpret a MarinSkyRL RL job's logs, metrics, and GPU state when judging its health.
Skills point here; this file owns the semantics. Cluster access and capture commands are in
`coreweave.md`.

## Progress truth

- **Phase timers are the progress truth** — `Started:`/`Finished: '<phase>'` lines
  (`load_checkpoints`, `generate`, `fwd_logprobs_values_reward`, `policy_train`,
  `sync_weights`, `wait_for_generation_buffer`) and the `Training Step Progress: N/M` counter.
  Not a trace count, not a progress bar, not engine heartbeats.
- Zero completed trials early in a long-episode agentic run is normal, not death. Judge
  freshness against the run's own cadence (a long forward can legitimately go quiet for over an
  hour at large scale).

## The colocated-engine deception (multi-mesh liveness)

vLLM engines and RolloutCoordinators are separate actors from the policy mesh. A hung policy
collective can present as `running` + fresh engine heartbeats + high average GPU util. The
signals that don't lie:

1. Ray actor-death lines in the per-actor logs (`worker-*.err`), not the finelog alone.
2. The NCCL watchdog: a real trip aborts the process; a lone watchdog log line does not.
3. The trainer/driver log advancing, with **policy-rank GPUs read separately from engine-rank
   GPUs** — never average across the two meshes.

Rank-0 logging trap: most worker diagnostics are gated to rank 0; "only rank 0 logged X" does
not mean only rank 0 ran X. The one deliberately ungated per-rank marker is
`WORKER_FORWARD_ENTER rank=N` — trust it for per-rank reachability, not the gated lines.

## Engine saturation — how to read it

- Point-in-time `nvidia-smi` SM-util reads high even on an idle engine. The trustworthy tuple:
  **`Waiting` queue depth + KV-cache usage + power draw + memory-bandwidth util** (from the
  vLLM scheduler log lines `Running: / Waiting: / GPU KV cache usage:`).
  - Saturated: `Waiting > 0`, KV off the floor, power near TDP.
  - Under-fed: `Waiting = 0` always, KV ≈ 0, power ≈ ⅓ TDP.
- **The sawtooth trough is usually benign backpressure.** Per-engine `Running` swinging
  peak→trough with `Waiting=0` while the run is `policy_train`-bound is the generation buffer's
  backpressure working. The one decider: `timing/wait_for_generation_buffer ≈ 0` means
  generation is off the critical path — do not "fix" it.
- **Engine under-subscription is a gen→dispatch→train pipeline problem, not a
  tools/sandbox-provider problem.** Measure while alive: are the RolloutCoordinator processes
  CPU/GIL-pegged (dispatcher-bound)? Is generation throttled by `max_staleness_steps`
  (training-bound)? Are `generate()` calls reaching engines that drain instantly (upstream
  dispatch rate)? A starvation verdict without these measurements is not a verdict.
- Concurrency is a lockstep tuple: `n_concurrent_trials` sets demand; decode capacity =
  `max_num_seqs × num_inference_engines` sets supply. Never permute one alone as a symptom fix.

## Rollout quality signatures

- Read actual trial artifacts (rewards, turns, agent output), never counts alone. Coherent
  multi-turn attempts at a low pass rate = learning; average ~1 turn per trial = a dead engine
  or broken loop.
- Incoherent token-salad output with 100% reward 0 after a weight sync = the FusedMoE w13
  reload-order fault; check the engine log for `initialize_layerwise_reload` /
  `finish_weight_reload` (the `SKYRL_W13_RELOAD_BRACKET` path, default on).
- OpenCode (installed agent, drives the model over HTTP from inside the sandbox) requires the
  controller ingress path — a cluster-internal `api_base` is unreachable from the sandbox and
  yields empty agent output, idle engines, and mass agent timeouts. In-process agents
  (terminus-2) are exempt.
- Sandbox idle-reap must be off for OpenCode (`auto_stop_interval_mins=0`): its model turns do
  not reset the idle timer, so queued trials get their sandbox reaped mid-trial
  (`Sandbox not found`, zero rollouts).
- 100% "failed to add tests directory" on a known-good dataset is a container problem (sandbox
  never created — usually dependency drift on the Daytona SDK path in a rebuilt image), not the
  dataset. Name the exception from a trial file you opened; never assume.
- A deterministic rollout `AttributeError` classified under `generate/errors/` almost always
  comes from a rollout dependency under the image's pins (an image/env issue), not first-party
  code — unless a verbatim traceback points at first-party files.

## Metrics — benign vs alarming

- `policy/rollout_train_prob_diff_mean` in the millions is a benign scale artifact (linear-space
  mean of exponentials; any fat tail dominates). The actual correctness read is the TIS family:
  `tis/imp_ratio_mean` near 1, `imp_ratio_capped_fraction` near 0, `log_ratio_abs_mean` a few
  hundredths of a nat, `generate/tis/exact_match_fraction` near 1, and a stable small
  `raw_grad_norm`. A broken router-replay instead shows `log_ratio_abs_max` in the tens,
  `policy_loss` ~1e4, `raw_grad_norm` ~1e5.
- Collapsing `policy_entropy` is a real warning sign; a missing metric on a fully-async path may
  just be a merge/reporting gap — check the source before alarming.

## Config-parse and resume signatures

- A run-YAML key not declared in `ppo_base_config.yaml` (at the launched ref) dies at
  config-parse: `Could not override '<key>' … not in struct` in the finelog, driver exit ~8–30 s
  after Ray attach, before any NCCL.
- Checkpoints nest at `<rundir>/<job_name>/checkpoints/global_step_N/` with
  `latest_ckpt_global_step.txt` beside them; the rundir top level proves nothing.
- With `resume_mode=latest` and chained restarts, `global_step` can overshoot `max_steps`; a run
  at or past its data-ceiling step count is complete, and errors in the overshoot tail are
  noise.

## Restart-burn rule

Check retries/restarts first: a run repeating the same failure on every attempt is
deterministically doomed (recommend kill + the fix); a restart burned on a genuine transient it
recovered from is benign. Get the count from `job summary` / the jobs table, and each prior
attempt's terminal error from the synced logs.
