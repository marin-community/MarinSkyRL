# Model initialization failure diagnostics

## Reported behavior

Policy workers remained inside Hugging Face `snapshot_download` for 66 minutes while the driver waited in `build_models` with an untimed `ray.get`. Three zero-byte `.incomplete` blobs did not change across three samples.

An independent OOM failure reached the head node as `RayTaskError(OutOfMemoryError)`. The fatal Loguru handler attempted to enqueue the exception object, raised `PicklingError`, and printed an unlevelled `Record was` diagnostic that an ERROR-level log filter omitted.

## Hypotheses

1. Model initialization has no wall-clock deadline, and its actor groups are assigned to the trainer only after initialization succeeds. A stalled download can therefore wait forever, and setup-time failure cannot use the trainer's normal actor cleanup.
2. `logger.opt(exception=True)` attaches the active dynamic Ray exception to the Loguru record. Worker logging uses `enqueue=True`, which requires the complete record to be pickleable.

## Reproduction

The CPU regressions model an unfinished Ray object reference and an exception whose `__reduce__` raises `PicklingError`. They require a 3,600-second shared initialization deadline, actor cleanup on expiry, an ERROR-level fatal message, and a fully pickleable Loguru record that contains the formatted traceback as text.

Initial result: test collection failed because neither the bounded initialization helper nor the safe fatal-log helper existed.

## Verification

The focused regression suite passes. It verifies that Ray receives the remaining 3,600-second initialization budget, timeout emits an ERROR message and kills the actors, the fully async trainer re-raises the original failure after teardown, and the resulting Loguru record contains no exception object and is pickleable.

The focused trainer and logging utility tests pass. The full CPU command used by PR CI also passes.
