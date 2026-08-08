# Debugging log for out-of-band HF export

Prevent long Hugging Face exports from blocking live training ranks or inheriting the training process group's collective timeout.

## Initial status

Two 67B FSDP2 exports failed after 1,800 seconds. Ranks 1–63 were waiting in the final `save_hf_model` barrier while rank 0 remained alive and continued serializing and uploading. A successful export took 1,720 seconds. Interrupted exports left incomplete object-store prefixes.

## Hypothesis 1

The final barrier is not needed for export correctness. The full-state-dict gather has completed before rank 0 enters serialization, and the Ray caller already waits for every worker result.

## Changes to make

Add a CPU regression around `FSDPStrategy.save_hf_model` that completes rank 0 serialization and fails if a post-gather barrier is entered. Remove that barrier while retaining the state-dict collective.

## Results

The regression initially failed during collection because no export-request module
existed. After the implementation, the strategy test completes serialization while a
patched trailing barrier raises if called.

## Hypothesis 2

Removing the barrier prevents this timeout but still blocks live training. Normal training should persist a rerunnable export request beside the completed sharded checkpoint; only an explicit export-only run should call `save_models`.

## Changes to make

Add a checkpoint-local request record, route normal HF-save callbacks to that record, and teach the export-only Iris command to execute and complete the request under its own job timeout. Validate that HF save intervals are checkpoint-aligned and protect checkpoints with pending export requests from retention cleanup.

## Results

Normal training now writes an atomic checkpoint-local request and never calls the live
rank export. The Iris export command consumes that request synchronously under its own
7,200-second default job timeout, marks success only after Iris exits zero, and restores
a failed request to pending so it can be rerun. Pending and unreadable requests protect
their source checkpoints from retention cleanup. Configuration validation rejects
misaligned checkpoint/export intervals and in-training Hub uploads that would otherwise
look for an export before it exists. Legacy Iris Hub destinations are carried in the
request and published by the export-only job after conversion.

The required launcher and trainer CPU suites passed: 1,266 tests passed and 20 were
skipped.

## Future work

- [ ] Add a controller-side service that discovers and schedules pending export requests automatically.
