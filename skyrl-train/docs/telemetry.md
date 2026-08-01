# MarinSkyRL telemetry

MarinSkyRL exports a small trainer critical-path slice directly through
`rigging.telemetry`. Install `skyrl-train[telemetry]`; Marin's GPU RL images
select this extra explicitly in their frozen `uv sync`. The default direct Iris
bootstrap installs no Marin package at runtime, while the controller-only
`marin-iris` bootstrap is conditional, so neither path is treated as an
implicit Rigging dependency. Export is inert unless all three of
`SKYRL_TELEMETRY_ENDPOINT`, `SKYRL_ROOT_RUN_UID`, and
`SKYRL_EXECUTION_UID` are set. `root_run_uid` stays fixed for the top-level RL
effort; `execution_uid` changes for every launch, resume, or retry. The service
is always `marinskyrl` and cannot be overridden. `SKYRL_SERVING_JOB_ID` is an
optional join to a centralized serving job.

The driver and trainer actor each own one exporter lifecycle. Export failures
are contained, shutdown uses one two-second deadline, and
telemetry never changes training results or W&B ownership. Before shutdown,
the adapter polls Rigging's public queued-record status only until that same
deadline so the pinned exporter's terminal record can leave a later batch.

## Current signals

| Name | Kind and unit | Attributes or body |
| --- | --- | --- |
| `work_completed` | counter delta, `{item}` | `work_kind=policy_step`, `role=trainer` |
| `phase_duration_seconds` | native histogram, `s` | `phase=rollout_or_inference_wait\|train_step`, `clock_domain=critical_path`, `outcome`, `role=trainer` |
| `progress_time_seconds` | gauge, `s` | last completed policy-step wall time |
| `policy_step` | gauge, `{step}` | latest completed policy step |
| `queue_depth`, `capacity` | gauges, `{item}` | bounded fully-async `rollout_buffer` only |
| `lifecycle`, `terminal` | structured events | process state; terminal status also includes the last progress, queue, and exporter-loss snapshot |

Resource attributes retain the configured run identities, `role`, `host`,
available Iris `job_id/task_id/attempt/worker/process_index`, and Ray-native
`ray_job_id/ray_task_id/ray_task_attempt/actor_uid/node_uid`. No generic
`run_id` or guessed GPU identity is emitted.

## Post-exit Finelog examples

Replace the timestamps and root identity before running these against
`telemetry_v1`. Keeping the literal millisecond bounds and entity predicate is
important for pruning.

Completed policy steps and their rate over the observed progress interval:

```sql
SELECT
  SUM(value) AS policy_steps,
  SUM(value) / NULLIF((MAX(timestamp_ms) - MIN(timestamp_ms)) / 3600000.0, 0) AS policy_steps_per_wall_hour
FROM "telemetry_v1"
WHERE service = 'marinskyrl'
  AND name = 'work_completed'
  AND json_get(attributes_json, 'work_kind') = 'policy_step'
  AND json_get(resource_attributes_json, 'root_run_uid') = 'ROOT_RUN_UID'
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-25 00:00:00') * 1000 AS BIGINT)
  AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-08-01 00:00:00') * 1000 AS BIGINT)
```

Exclusive trainer rollout/inference wait fraction:

```sql
SELECT
  SUM(CASE WHEN json_get(attributes_json, 'phase') = 'rollout_or_inference_wait' THEN value ELSE 0 END)
    / NULLIF(SUM(value), 0) AS rollout_or_inference_wait_fraction
FROM "telemetry_v1"
WHERE service = 'marinskyrl'
  AND name = 'phase_duration_seconds'
  AND json_get(attributes_json, 'clock_domain') = 'critical_path'
  AND json_get(resource_attributes_json, 'execution_uid') = 'EXECUTION_UID'
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-25 00:00:00') * 1000 AS BIGINT)
  AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-08-01 00:00:00') * 1000 AS BIGINT)
```

Policy progress and freshness at exit:

```sql
SELECT
  MAX(CASE WHEN name = 'policy_step' THEN value END) AS policy_step,
  MAX(CASE WHEN name = 'progress_time_seconds' THEN value END) AS last_progress_time_seconds,
  EXTRACT(EPOCH FROM TIMESTAMP '2026-08-01 00:00:00')
    - MAX(CASE WHEN name = 'progress_time_seconds' THEN value END) AS progress_age_seconds
FROM "telemetry_v1"
WHERE service = 'marinskyrl'
  AND name IN ('policy_step', 'progress_time_seconds')
  AND json_get(resource_attributes_json, 'execution_uid') = 'EXECUTION_UID'
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-25 00:00:00') * 1000 AS BIGINT)
  AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-08-01 00:00:00') * 1000 AS BIGINT)
```

Failure timeline with actor attempt, last progress, rollout queue, exporter
loss, and terminal reason:

```sql
SELECT
  to_timestamp_millis(timestamp_ms) AS observed_at,
  json_get(resource_attributes_json, 'role') AS role,
  json_get(resource_attributes_json, 'actor_uid') AS actor_uid,
  json_get(resource_attributes_json, 'ray_task_attempt') AS actor_attempt,
  json_get(body_json, 'policy_step') AS policy_step,
  json_get(body_json, 'last_progress_time_seconds') AS last_progress_time_seconds,
  json_get(body_json, 'queue_depth') AS queue_depth,
  json_get(body_json, 'queue_capacity') AS queue_capacity,
  json_get(body_json, 'export_lost_records') AS export_lost_records,
  json_get(body_json, 'status') AS status,
  json_get(body_json, 'reason') AS terminal_reason
FROM "telemetry_v1"
WHERE service = 'marinskyrl'
  AND name = 'terminal'
  AND json_get(resource_attributes_json, 'root_run_uid') = 'ROOT_RUN_UID'
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-07-25 00:00:00') * 1000 AS BIGINT)
  AND timestamp_ms < CAST(EXTRACT(EPOCH FROM TIMESTAMP '2026-08-01 00:00:00') * 1000 AS BIGINT)
ORDER BY timestamp_ms
```

## Deferred work

- Rollout, sample, and generated-token accounting; worker inference queue versus
  compute; environment/reward phases; weight fetch/decode/apply and policy lag.
- Routing and load imbalance, session affinity, fallback and dead-engine
  signals, and exhaustive worker/engine role instrumentation.
- Fabric and hardware probes. Iris, DCGM, and Ray remain authoritative where
  they already cover a source. Rigging NVIDIA/NCCL probes remain disabled until
  a dependency-resolvable published Rigging version contains Marin commit
  [`805dd5e47`](https://github.com/marin-community/marin/commit/805dd5e47483e249dd63f12d3da6cf8cd6a29a12).
  The telemetry extra is pinned to
  the earlier base-exporter release `marin-rigging==0.2.67.dev30673006835`, so
  MarinSkyRL does not copy or activate the newer probe implementation. SkyRL
  imports only `rigging.telemetry` and `rigging.timing`; lock metadata excludes
  the wheel's unrelated Connect dependency because its OpenTelemetry floor is
  incompatible with the Megatron extra's ceiling.
