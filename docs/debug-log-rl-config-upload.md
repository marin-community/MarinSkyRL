# Debugging log for out-of-tree RL configs

Make `iris_backend.py` accept a readable absolute RL config outside the repository without shipping a stale duplicate.

## Initial status

`parse_rl_config` accepts any readable absolute path, but launcher normalization rejects paths outside `PROJECT_ROOT` before submission. The CLI help claims absolute paths are supported.

## Hypothesis 1

The launcher can preserve host-side parsing of the source file while passing its validated bytes through the Iris task environment and materializing an in-pod copy before the controller starts.

## Changes to make

Add a regression test that normalizes an out-of-tree config and observes the resulting task command. Then add explicit host and in-pod config paths plus a task bootstrap that writes the captured bytes without exposing them in the printed command.

## Results

Before the fix, the focused tests failed in both reported ways: an out-of-tree absolute path raised
`SystemExit`, and a missing relative path emitted a warning without failing. After the change, both
tests pass. The task command references a content-addressed path under `/tmp/marin-rl-configs` and
does not contain the launch host path. The full `cloud/iris/tests/` suite passes with 128 tests and
2 skips.

## Hypothesis 2

The initial fix duplicated shared config resolution and spread delivery state across dynamic
`argparse.Namespace` attributes, which could let host and task behavior drift.

## Changes to make

Reuse `resolve_rl_config_path`, represent delivery with a frozen `RlConfigLaunch`, require
normalization before command construction, and derive the submitted task environment from that
record.

## Results

The focused regressions still pass after the refactor. The payload mapping and shell reference use
the same `RL_CONFIG_PAYLOAD_ENV` constant, and config-defined environment values cannot replace the
launcher-owned payload.

## Hypothesis 3

Materializing the config in launcher-generated shell duplicates application behavior and makes the
delivery contract difficult to test end to end. The in-container Python runner is the natural owner
of turning the forwarded payload back into a file before parsing it.

## Changes to make

Move payload decoding and file creation into `rl_config_translation.py`, call it from `training_driver.py`,
and have the regression test pass the launcher's environment mapping through that worker helper.

## Results

The focused launcher and runner tests pass. The regression now verifies that the worker writes the
exact source bytes from the environment produced by normalization, while the task command contains
only the content-addressed task path and never the launch-host path.

## Future work

- [x] Record the failing regression result and final test results.
