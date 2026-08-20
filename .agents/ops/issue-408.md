# Debugging log for MarinSkyRL issue 408

Determine why scheduled Actions run 32354323948 failed and restore the nightly lane.

## Initial status

The `gsm8k-h100` job reached training on Iris and generated its first rollout, then failed in
`TrajectorySink._select_archive` with `ValueError: trajectory retention archive overhead exceeded its reserved
bound`. The parallel `grug-vllm-gb200` job passed. The checked-out worktree predates the failing run's main-branch
revision, so the failing revision must be inspected before reproducing or changing code.

## Hypothesis 1

A recent trajectory-retention change underestimates archive serialization overhead for the nightly's real first
batch, causing the internal byte-budget invariant to reject otherwise valid retained trajectories.

## Changes to make

Inspect the failing revision and its retention tests, then reproduce the selector failure with the smallest real
batch shape. Do not change runtime behavior until the accounting mismatch is identified.

## Results

Confirmed. A regression test that sends 256 mandatory trajectories through `TrajectorySink.retain` fails on
unmodified `main` with the nightly's `trajectory retention archive overhead exceeded its reserved bound`
exception. All 256 records pass selection before the exception. The existing three-record tests do not exhaust
the 2 KiB base allowance that masks the per-record underestimate.

## Hypothesis 2

Computing the deterministic ZIP and manifest size from the selected records will preserve the configured hard
byte bounds without a batch-size-dependent allowance.

## Changes to make

Replace the fixed base and per-record allowances with exact ZIP_STORED layout accounting shared with manifest
construction. Keep the final encoded-size assertion, then run the 256-record regression and the full retention
test module.

## Results

The exact layout calculation accounts for each stored payload, local file header, central-directory header,
filename, manifest payload, and end record. The 256-record regression passes and persists one archive containing
all records. All 16 trajectory-retention tests pass, including byte bounds, restart reconciliation, publication
timeouts, and best-effort backpressure. The broader trajectory-runner suite passes 249 tests with one skip when
run with access to its Hugging Face tokenizer fixtures; its first sandboxed attempt failed only at those fixture
downloads. Changed-file Ruff checks and formatting pass.

The advisory branch review found a stale design-doc sentence about reserved overhead and duplicated record-entry
name construction across the manifest, size projection, and writer. The documentation now describes exact
layout accounting, and one helper defines the entry name for all three paths.

## Future work

- [x] Add a regression test for the nightly's 256-trajectory batch shape.
- [x] Run the full trajectory-retention module and broader trajectory-runner CPU suite.
