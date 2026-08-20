# Training trajectory retention

MarinSkyRL retains a bounded sample of normalized training trajectories for debugging, replay, and audit. Retention runs at the shared `TrajectoryRunner` output boundary, after reward shaping and alignment metrics. SkyRL Gym, StepWise, Terminal Bench, synchronous training, fully asynchronous training, and evaluation therefore use one record contract.

## Ownership and lifecycle

The trainer driver owns one `TrajectorySink` and binds it to the selected trajectory runner. The driver builds and selects records so one ledger governs the entire run. Storage operations run in a spawned child process with a hard deadline. A wedged client can therefore be terminated without leaving a Python thread holding the sink lock.

Best-effort publication has one bounded pending slot. Finalization enqueues an archive and returns without waiting for object storage. If the preceding archive is still pending, the sink drops the new retention batch and reports backpressure instead of delaying training. Required publication waits for the same isolated operation and fails within `publish_timeout_seconds`. Shutdown gives a pending best-effort archive `shutdown_timeout_seconds` to finish before terminating its worker.

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

Records are canonical JSON compressed with deterministic gzip settings. One finalized runner batch is stored as a deterministic ZIP archive containing the selected `.json.gz` records and a manifest. This replaces hundreds of serial object-store PUTs with one archive PUT plus one ledger PUT. The sink checks the encoded archive against the remaining per-step and per-run budgets. If the desired archive is too large, it re-encodes the largest deterministic-priority prefix that fits before publication. Content-derived record and archive IDs make resumed or retried writes idempotent.

The archive is written before the ledger. Initialization lists stored objects and reconciles archives or legacy individual records that are missing from the ledger, so an interrupted ledger update cannot orphan byte accounting. Deterministic count sampling remains logical: superseded count-only records can remain inside an append-only archive, but the ledger removes their selection reason and physical archive bytes continue to count against both bounds.

Configured dotted fields are redacted before the record ID is calculated and before content reaches storage. A record path includes the fixed schema version, phase, global step, hashed instance identity, repetition, and record ID.

## Failure contract

With `required: false`, publication is asynchronous. Enqueue, completion, timeout, error, and backpressure metrics are emitted under `generate/trajectory_retention/*`; completion metrics can appear on the next generated batch. Training continues after a timeout or storage error and reconciles storage before accepting another archive. With `required: true`, publication succeeds before finalization returns or raises a bounded error that names the archive. A disabled sink or a phase outside the configured set performs no work and emits no retention metrics.

## Regression contract

CPU tests cover the complete normalized schema, the real multi-row StepWise shape, training capture independent of validation, mandatory anomaly selection, order-independent asynchronous sampling, compressed-byte limits, redaction, replay provenance, resume idempotence, asynchronous best-effort finalization, required failures, hard worker termination, and reconciliation after an archive write without a ledger commit. The generator-interface test ensures new generator subclasses inherit retention without implementing a private hook.
