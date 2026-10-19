# Evaluation scheduling

An evaluation sweep pauses the training producer and drains its submitted generation
requests for up to `trainer.eval_drain_timeout_seconds` (300 seconds by default).
At the deadline it revokes unfinished groups and cancels their remote requests.
Groups committed before revocation remain available for training; late writes from revoked groups are
ignored. Cancellation cleanup has a separate per-request limit,
`trajectory_runner.rollout_workers.cancellation_timeout_seconds` (120 seconds by
default), with all cancellations issued concurrently. Cleanup failure fails the run.
Discarding unfinished groups can bias retained training data toward faster tasks; monitor
`eval/drain_discarded_groups`, the number of revoked groups in each sweep.

When inference already has the evaluated policy version, evaluation requests can be
submitted while training requests drain. Execution still shares the configured worker
capacity with those training requests. The ordinary training
step synchronizes weights before its evaluation callback, so it uses this overlap.
If evaluation needs a different policy version, it first awaits the bounded drain
and synchronizes weights. The optimizer cannot advance until both evaluation and
drain finish. Draining does not consume completed groups or perform optimizer updates.
Training submissions resume when the evaluation session exits, including on errors.

Native validation rows, named sampling profiles, and inline Harbor trials enter one
bounded evaluation pool. Its total capacity is `trajectory_runner.max_concurrent_tasks`,
the individual-task execution capacity. Training's `max_in_flight` counts multi-sample
prompt groups and independently bounds training prefetch. Free slots are filled
continuously, round-robin across sources with pending work. Native requests use the
least-loaded rollout worker, just as training requests do. `trainer.eval_batch_size`
controls dataloader reads, not a generation barrier. A slow task no longer holds up
the next block of validation rows.

Each Harbor cohort's `job.n_concurrent_trials` is a source cap inside the global pool.
Idle slots are available to other sources. A single subprocess receives admitted
trials over a private socket. The trainer reads the named evaluation credential and
sets `DAYTONA_API_KEY` only in the child environment; it does not replace the training
process's `DAYTONA_API_KEY`. It runs the upstream Harbor `Job` lifecycle for each
task, including retries, verification and artifact publication. Per-task job names
append `-task-00000`, `-task-00001`, etc. to the step/cohort job name. The callback
aggregates their results into the original cohort metrics.

Evaluation runs every selected row and repetition without training admission filters
or reward-based replacement. Trace instance IDs include the evaluation step and a
length-prefixed sampling-profile name; named profiles cannot overwrite each other's
traces. Metrics still group repetitions by the original task UID within each profile.
Failed and unscored trials remain in attempted counts;
coverage determines checkpoint-selection eligibility. Native batches never enter the
training buffer. Harbor configurations require unique pinned tasks, one agent, and
one attempt per task. Retries remain explicitly configured Harbor infrastructure
retries. The full sweep, including all cohorts, uses one policy version.

Inline Harbor scoring retains verifier rewards after agent-attributable stops and
partial-work terminations, as classified by Harbor's pinned error taxonomy.
Infrastructure and unknown errors remain unscored, even if a reward is present.
Results without a verifier reward remain unscored for every category. Scoring uses
the verifier reward without the training loss's error-treatment overrides.

Callbacks that add evaluation sources implement `collect_evaluation(state, trainer,
pool)` and admit each unit of work with `pool.run(source, run, limit=...)`. They return
metrics after all requests complete. `on_evaluate` remains the ordered reduction and
checkpoint-selection phase, after source collection has finished. Exceptions in
collection fail the sweep and cancel remaining work rather than reporting partial
results as a successful evaluation.

The trainer reports `timing/eval_drain` separately from evaluation collection so a
long tail in outstanding training generation can be distinguished from slow evals.
Session logs report dispatched request counts for every rollout worker. The admission
summary records each source's started and finished counts and first-start/last-finish
offsets in seconds, peak total active requests, and whether requests from different
sources were active simultaneously. These summaries contain no task content.

Per-task evaluation JSON records retain failure categories and safe fields under
`verification.diagnostics`: `operation`, `machine_operation`, `timed_out`,
`provider_status_code`, `provider_sdk_call`, and `provider_rate_limits`. SDK method
names are allowlisted, including generated request methods that distinguish sandbox
creation from toolbox URL lookup. A compound SDK method alone does not identify
its failing HTTP request. Rate-limit diagnostics retain only nonnegative integer
values from named Daytona limit, remaining, reset, and retry-after headers, without
unit conversion. `DAYTONA_RATE_LIMIT_HEADERS` in
[the task worker](../skyrl-train/skyrl_train/rollouts/task_worker.py) lists the retained keys.
Raw provider messages, other headers, arbitrary error codes, and verifier
references are excluded. An unscored task remains unscored even when its failure can
be classified; diagnostic fields do not supply a reward or change retry behavior.
