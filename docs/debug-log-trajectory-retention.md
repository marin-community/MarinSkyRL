# Debugging log for unified trajectory retention

Retain bounded, replayable training trajectories after every generator has produced the normalized `GeneratorOutput` contract.

## Initial status

Training outputs reach the driver in both synchronous and fully asynchronous trainers, but only validation has a shared generation logger. TerminalBench persists its own Harbor trials; SkyRL Gym and Verifiers have no durable training sink.

## Hypothesis 1

A trainer-owned sink can cover every generator without duplicating retention budgets across async workers or adding decoded text to Ray's object-store payloads.

## Red contract

The initial retention test failed during collection because `skyrl_train.generators.trajectory_retention` did not exist. The contracts require a common schema, validation-independent training retention, deterministic bounds, idempotent resume, replay provenance, and explicit required versus best-effort failure behavior.

## Changes to make

Build records from `GeneratorInput` plus normalized `GeneratorOutput`, then invoke one sink from synchronous generation, fully async generation completion, and evaluation. Use content-addressed compressed records and a persistent run ledger so retries do not overwrite or duplicate trajectories.

## Results

The sink is owned by the trainer driver and attached to `GeneratorInterface`, whose shared output-finalization hook covers synchronous, fully asynchronous, and evaluation generation. A persistent ledger maintains deterministic per-step top-k sampling even when async groups complete in different orders. Generator-specific rows are adapted only through the normalized contract; StepWise rows are grouped into one record with explicit token and termination boundaries.

Focused tests cover the complete normalized schema, a real multi-row StepWise shape, validation-independent training capture, anomaly selection, order-independent async sampling, compressed-byte bounds, replay provenance, idempotent resume, and required versus best-effort storage failure.

## Review

The lint-review pass found duplicated path derivation, sink construction, atomic-write logic, stop-reason defaults, per-call retention hooks, and untyped records. The implementation now has one launcher path helper, one sink factory, one parent-class finalization hook, one shared local/cloud atomic-write primitive, a shared configurable stop contract, and typed trajectory, ledger, and row-group records. Best-effort storage failures log the concrete error as well as emitting metrics. The normalized-schema test covers the common boundary directly, while the separate StepWise contract exercises its actual multi-row shape.
