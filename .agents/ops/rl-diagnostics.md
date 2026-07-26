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

## Training dynamics

Per-step training metrics stream into the job's finelog (and wandb when the run has a
tracker). Judge every trend over a window of ≥10 steps from the synced files — agentic RL
rewards are noisy and long-episode arms bank slowly.

- **Reward curve:** all-zero from step 1 is an infra fingerprint (over-quota Daytona org,
  sandbox reap, broken weight sync, missing task data), not "the model is bad" — cross-check
  the rollout signatures above. A collapse from a previously-positive plateau is a regression
  event: find the step and correlate with a weight-sync / resume / config event.
  Noisy-but-drifting-up is healthy.
- **Entropy:** `policy_entropy` collapsing toward 0 early (rewards not improving) or an
  abrupt mid-run cliff is a real warning sign; slow decline alongside improving reward is
  expected. A missing metric on a fully-async path may just be a merge/reporting gap — check
  the source before alarming.
- **TIS family, healthy on-policy references:** `tis/imp_ratio_mean` ≈ 0.79–1.02,
  `tis/log_ratio_abs_mean` ≈ 0.06–0.094 nats (the inherent vLLM↔trainer bf16 precision gap —
  what TIS corrects, not misalignment), `tis/imp_ratio_capped_fraction` ≈ 0–1e-4,
  `generate/tis/exact_match_fraction` ≈ 0.99, and a stable small `raw_grad_norm`.
- **Broken router-replay fingerprint:** `log_ratio_abs_max` ~19, `policy_loss` ~1e4,
  `raw_grad_norm` ~1e5. Sustained ratio drift far from 1, a growing capped fraction, or a
  log-ratio step change right after a weight sync means the engines are serving stale or
  wrong weights — cross-check the w13 reload bracket (§Rollout quality signatures) and
  `generate/tis/lcs_fallback_fraction` (a spike = target corruption).
- Both backends emit the `tis/imp_ratio_*` family from main @ `d7ba00ff` onward. A run
  launched from an older bundle lacks them on the Megatron backend — judge those runs by
  the `generate/tis/*` fractions plus the `policy_loss` / `raw_grad_norm` magnitudes.
- `policy/rollout_train_prob_diff_mean` in the millions is a benign scale artifact
  (linear-space mean of exponentials; any fat tail dominates) — never read it as a fault.
- **Step time:** `timing/wait_for_generation_buffer` ≈ 0 means generation is off the
  critical path (§Engine saturation). Tabulate the per-phase Timer lines across the window;
  one phase inflating step-over-step while its mesh's GPUs sit low is a pipeline bubble —
  attribute it with the gen→dispatch→train measurements in §Engine saturation, not a guess.

## Per-trial duty cycle (TimingInfo)

- `result.json` count = trials actually finished; `config.json`-only dirs are
  started/unfinished. Access pattern and layout: `coreweave.md` §Trial artifacts.
- Each trial's `result.json` carries per-phase `TimingInfo {started_at, finished_at}`:
  `environment_setup` (sandbox create), `agent_setup`, `agent_execution` (LLM-gen +
  tool-exec combined), `verifier`. `agent_result.metadata.api_request_times_msec` is the
  LLM-gen-only per-call list. Derived: tool-exec = `agent_execution` − LLM-gen; teardown gap
  = trial `finished_at` − max(phase `finished_at`), sub-second when healthy.
- Aggregate bounded samples (newest ~200 for steady state, ~500 for error tails) and report
  medians/percentiles only.
- LLM-gen ≫ sandbox (e.g. ~89% vs <1%) = LLM-turn-bound; the lever is trial concurrency /
  buffer depth, not sandbox optimization. Sandbox create/teardown above ~10% of wall, or an
  `environment_setup` tail over 10 s, is real re-provision churn.
- Burst ≠ churn: a wave of `environment_setup` time at job start or right after a
  preempt/relaunch is initial provisioning, not churn.

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
