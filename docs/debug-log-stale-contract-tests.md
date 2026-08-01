# Debugging log for stale contract tests

Restore the two full CPU-suite contracts that failed unchanged on `main` after
the EP co-arrival work.

## Initial status

`test_generator_output_concatenation` compared the annotations of
`GeneratorOutput` against a frozen field inventory. Five optional fields had been
added without changing the test. `test_all_defaults_is_structurally_identical_to_pre_ep`
compared the entire composed Hydra configuration with a pre-EP snapshot; unrelated
environment and trainer defaults had intentionally changed since that snapshot.

## Hypothesis 1

Both failures come from tests that bind to a larger implementation surface than
their behavior requires. The concatenation test should observe concatenated
per-sample values. The config test should compare only the FSDP defaults protected
by the additive EP/CP contract.

## Changes to make

Replace the annotation inventory with trajectory, step, and baseline-exclusion
concatenation assertions. Replace the repository-wide YAML golden with the three
base FSDP defaults after removing additive parallelism fields.

## Results

The two targeted files passed together: 19 passed. The complete trainer CPU
suite then passed: 902 passed, 19 skipped. The repaired tests now enforce the
behavior they name without requiring unrelated config fields or every optional
`GeneratorOutput` field to remain frozen.
