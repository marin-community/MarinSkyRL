# MarinSkyRL telemetry

Install `skyrl-train[telemetry]` to export driver and trainer lifecycle, policy
steps, generated rollouts, samples and tokens, exclusive rollout or inference
wait and train-step durations, and fully async rollout-buffer occupancy through
`rigging.telemetry`. The same extra lets each Iris controller forward a bounded
allowlist of its local Ray scheduler, logical CPU/GPU, placement-group and object
store snapshots. A rollout is one completed trajectory; a sample is one generated
response segment, so step-wise training counts only terminal segments as
rollouts. Export is inert unless `SKYRL_TELEMETRY_ENDPOINT` and `SKYRL_RUN_ID`
are set. `cloud/iris/telemetry_env.py` writes them from the
task's own Iris context — the log server resolves to a pod-reachable address only
from inside the cluster — and the task runtime exports them before Ray starts, so
the raylet and its actors inherit them. `SKYRL_EXECUTION_UID` identifies one task
attempt and each process derives its own from `IRIS_TASK_ID` when it is unset,
because an attempt spans several pods. The service is fixed to `marinskyrl`;
`SKYRL_SERVING_JOB_ID` optionally joins a centralized serving job.

Each row's resource carries `run_id`, which Finelog promotes to the `telemetry_v1`
column of that name and every Grafana panel joins on. It defaults to the Iris job
id; `--run-id` sets the experiment identity the caller holds instead. marin#8379
renamed this attribute from `root_run_uid`, which has no column.

Export and shutdown failures do not change training results or W&B ownership.
The Ray allowlist discards worker, address and task-name labels and never forwards
Ray's physical node or GPU families; Iris remains authoritative for host and GPU
telemetry, while centralized vLLM metrics stay with the serving job. Hardware
probes are not started. The frozen GPU runtime profile selects the telemetry
extra, and process shutdown gives Rigging at most two seconds to drain queued
records.
