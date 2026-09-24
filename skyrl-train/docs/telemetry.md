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
runtime exports them before Ray starts so its actors inherit them. Rigging also
discards records from a process that never configured it, so the trainer and
driver configure it in the entrypoint and every worker actor configures it in its
constructor.
`SKYRL_EXECUTION_UID` can override the execution identity; otherwise each process
uses its node-local `IRIS_ATTEMPT_UID`. The service is fixed to `marinskyrl`;
`SKYRL_SERVING_JOB_ID` optionally joins a centralized serving job.

Each row's resource carries `run_id`, which Finelog promotes to the column of the
same name; the launch document's `run.id` sets it. Rows also carry
`training_type`, `sync` or `async`, from the launch document's
`runtime.training_type`, which the task runtime exports as `SKYRL_TRAINING_TYPE`.

`docs/design/async-rl-telemetry.md` at the repository root lists every metric
family, the question it answers, its switch, default and cost, and the panels
that read it.

Export and shutdown failures do not change training results or W&B ownership.
The Ray allowlist discards worker, address and task-name labels and never forwards
Ray's physical node or GPU families; Iris remains authoritative for host and GPU
telemetry, while centralized vLLM metrics stay with the serving job. Hardware
probes are not started. The frozen GPU runtime profile selects the telemetry
extra, and process shutdown gives Rigging at most two seconds to drain queued
records.
