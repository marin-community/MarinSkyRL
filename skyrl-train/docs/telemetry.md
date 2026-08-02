# MarinSkyRL telemetry

Install `skyrl-train[telemetry]` to export the driver and trainer lifecycle,
completed policy steps, generated rollouts, samples and tokens, exclusive
rollout or inference wait and train-step durations, and fully async
rollout-buffer occupancy through `rigging.telemetry`. A rollout is one completed
trajectory; a sample is one generated response segment, so step-wise training
counts only terminal segments as rollouts. Export is inert unless
`SKYRL_TELEMETRY_ENDPOINT`, `SKYRL_ROOT_RUN_UID`, and `SKYRL_EXECUTION_UID` are
set; the service is fixed to `marinskyrl`, and `SKYRL_SERVING_JOB_ID` optionally
joins a centralized serving job. Marin's GPU images install the pinned extra
explicitly because neither the direct Iris launcher nor its conditional
controller bootstrap otherwise guarantees Rigging.

Export and shutdown failures do not change training results or W&B ownership.
Ray, vLLM, DCGM, and Iris remain authoritative for metrics they already expose,
and hardware probes stay disabled until a published Rigging version contains
[Marin commit `805dd5e47`](https://github.com/marin-community/marin/commit/805dd5e47483e249dd63f12d3da6cf8cd6a29a12);
the current pin is `marin-rigging==0.2.67.dev30673006835`. Shutdown polls only
Rigging's public queued-record status within the existing two-second deadline
before using the remaining budget for `shutdown()`.
