# MarinSkyRL telemetry

Install `skyrl-train[telemetry]` to export driver and trainer lifecycle, policy
steps, generated rollouts, samples and tokens, exclusive rollout or inference
wait and train-step durations, and fully async rollout-buffer occupancy through
`rigging.telemetry`. The same extra lets each Iris controller forward a bounded
allowlist of its local Ray scheduler, logical CPU/GPU, placement-group and object
store snapshots. A rollout is one completed trajectory; a sample is one generated
response segment, so step-wise training counts only terminal segments as
rollouts. Export is inert unless `SKYRL_TELEMETRY_ENDPOINT`,
`SKYRL_ROOT_RUN_UID`, and `SKYRL_EXECUTION_UID` are set. The service is fixed to
`marinskyrl`; `SKYRL_SERVING_JOB_ID` optionally joins a centralized serving job.

Export and shutdown failures do not change training results or W&B ownership.
The Ray allowlist discards worker, address and task-name labels and never forwards
Ray's physical node or GPU families; Iris remains authoritative for host and GPU
telemetry, while centralized vLLM metrics stay with the serving job. Hardware
probes are not started. The GPU images select the telemetry extra, and process
shutdown polls Rigging's public queue status for at most two seconds.
