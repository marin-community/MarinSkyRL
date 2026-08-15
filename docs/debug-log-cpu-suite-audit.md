# Debugging log for CPU suite audit

Audit the CPU failures above `weaver/fix-hydra-storage-overrides`, remove only invalid tests, and preserve the
lifecycle storage fix and meaningful behavior coverage.

## Initial status

The documented CPU environment collected 1,463 tests. A restricted run reported Gloo interface, dataset
multiprocessing, repository comment-policy, and Jupiter report failures before stalling in a distributed test.
The three lifecycle storage regression tests passed.

## Hypothesis 1

The Gloo and dataset failures are sandbox restrictions, not obsolete tests.

## Changes to make

Run those files outside the restricted sandbox with the frozen CPU dependency profile.

## Results

All five Gloo tests and all eight dataset tests passed outside the sandbox. Their coverage remains unchanged.

## Hypothesis 2

`test_no_comments.py` is an invalid style gate rather than a behavior test.

## Changes to make

Remove the file after checking its history and the current repository testing and comment policies.

## Results

The file came from PR #165 and enforces zero YAML and TOML comments plus a two-line limit on Python comment
blocks. The current Marin rules allow comments that explain subtle behavior and put lint in `infra/pre-commit.py`.
Its Python check also depends on a local branch named `main`, silently skips when that ref is absent, and scanned
124 stale commits in this worktree because local `main` lagged `origin/main`.

## Hypothesis 3

The Jupiter failure is one invalid assertion inside a meaningful end-to-end test.

## Changes to make

Remove the assertion against a rendered Markdown substring while retaining the structured report and artifact
assertions and the guard that rejects Iris trace synchronization for Jupiter jobs.

## Results

The Markdown table wraps cells to the detected terminal width, splitting `TIS exact=0.99` in an 80-column PTY.
The focused `report_row` test already covers the exact TIS trend value. The end-to-end test continues to cover
Jupiter report publication and all backend-specific artifact fields through `latest.json`.

## Validation

- The focused Iris tests passed: 17 passed, including all three lifecycle storage regression tests.
- The complete Iris CPU suite passed on the current pull request head: 306 passed. Pytest initially loaded the
  pre-rebase `marinskyrl` package from the virtual environment; rebuilding the local package made the new
  quoted-lifecycle-path regression use the current source, where it passed without changes.
- The trainer CPU suite ran without assertion failures until the process was killed with exit 137 at
  `test_build_dataloader_seeding` after cumulative memory growth. That test passed alone, and the remaining 151
  tests passed in a separate run. The split runs preserve every selection and deselection from the CI command.
- `infra/pre-commit.py --changed-files --fix` passed Ruff check and format.

## Future work

- [ ] None.
