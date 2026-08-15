# Training trajectory retention

MarinSkyRL retains a bounded sample of normalized training trajectories for debugging, replay, and audit. Retention runs at the shared `TrajectoryRunner` output boundary, after reward shaping and alignment metrics. SkyRL Gym, StepWise, Terminal Bench, synchronous training, fully asynchronous training, and evaluation therefore use one record contract.

## Ownership and lifecycle

The trainer driver owns one `TrajectorySink` and binds it to the selected trajectory runner. The sink runs on the driver rather than Ray workers so one ledger governs the entire run. Writes run in a thread to avoid blocking the trainer event loop.

Each record combines the normalized `TrajectoryRequestBatch` and `TrajectoryBatch`:

- trajectory identity, source rows, environment class, and environment extras;
- prompt messages and token IDs;
- response text, token IDs, loss mask, trainable spans, step boundaries, stop reason, and loop spans;
- raw outcome reward, shaped reward, and shaping components;
- rollout metrics;
- runner, inference backend, model, checkpoint, model step, sampling parameters, and shaping schema provenance.

StepWise rows with the same trajectory ID become one record. Explicit row and token boundaries preserve the individual generation steps.

## Selection and bounds

The default policy retains one deterministic sample per training step plus failures, non-terminating responses, and detected loops. Operators can add a fractional sample or reward thresholds. Count sampling uses a stable SHA-256 score and a persistent top-k ledger, so fully asynchronous completion order does not affect the selected set.

Records are canonical JSON compressed with deterministic gzip settings. The sink checks compressed bytes against per-step and per-run budgets before writing. Content-derived record IDs and the persistent ledger make resumed or retried writes idempotent. Local paths and S3 or GCS URLs use the same atomic I/O contract.

Configured dotted fields are redacted before the record ID is calculated and before content reaches storage. A record path includes the fixed schema version, phase, global step, hashed instance identity, repetition, and record ID.

## Failure contract

With `required: false`, storage or ledger errors are logged and emitted under `generate/trajectory_retention/*`; training continues. With `required: true`, the original error propagates and fails the run. A disabled sink or a phase outside the configured set performs no work and emits no retention metrics.

## Regression contract

CPU tests cover the complete normalized schema, the real multi-row StepWise shape, training capture independent of validation, mandatory anomaly selection, order-independent asynchronous sampling, compressed-byte limits, redaction, replay provenance, resume idempotence, and both storage failure modes. The generator-interface test ensures new generator subclasses inherit retention without implementing a private hook.
