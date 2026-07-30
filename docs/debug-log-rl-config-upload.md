# Debugging log for out-of-tree RL configs

Make `launch_rl_iris.py` accept a readable absolute RL config outside the repository without shipping a stale duplicate.

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

## Future work

- [x] Record the failing regression result and final test results.
