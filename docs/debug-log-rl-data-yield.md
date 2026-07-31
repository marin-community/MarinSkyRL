# Debugging log for RL data conversion yield

Allow reproducible dataset preparation to skip malformed source rows with warnings and
provenance, while offering an opt-in minimum conversion-yield gate.

## Initial status

`prepare_artifact` calls each source transform directly. A `ValueError` or `TypeError` from
one malformed source row aborts the complete artifact before provenance can be written.

## Hypothesis 1

Row-local conversion and validation failures can be isolated without hiding programmer
errors by catching only `ValueError` and `TypeError`, recording their indices and messages,
and applying a minimum-yield check after the scan.

## Changes to make

Write regression tests for a majority-malformed source that succeeds by default and for
the same source failing an explicit minimum-yield gate. Then add structured provenance,
a visible warning, and the CLI option.

## Results

Regression coverage establishes two behaviors. Majority-malformed input emits a warning
and produces every valid row with failure details in provenance. The same input fails when
a 50% minimum yield is requested. All-invalid inputs remain hard failures and retain the
first row-level diagnostic cause.

## Future work

- [ ] Consider a bounded sidecar for per-row diagnostics if very large malformed datasets make provenance unwieldy.
