# Debugging log for registered RL entrypoints

Prevent an RL config from sending an arbitrary or deleted Python module to an Iris allocation.

## Initial status

`parse_rl_config()` returns the top-level `entrypoint` string unchanged. The Iris launcher's `normalize()` function
uploads external configuration bytes without inspecting that value, including during `--dry-run`.

## Hypothesis 1

A shared symbolic-name resolver at the RL configuration boundary can reject stale module paths before submission and
keep the task-side driver on the same contract.

## Changes to make

- Add submit-path coverage for an external config containing the deleted Terminal-Bench module path.
- Add parse coverage for resolving the symbolic `terminal_bench` name.
- Replace module paths in repository-owned RL configs with symbolic names.
- Resolve and validate the name in both launcher normalization and task-side parsing.

## Results

The external-config dry-run test failed before the implementation because `normalize()` accepted the deleted module
path. The named-resolution test also showed that task-side parsing returned the YAML string unchanged.

The shared resolver now accepts only registered names and verifies that the mapped package module exists. Launcher
normalization applies it before constructing the task payload; task-side parsing uses the same resolver. Both regression
tests and the Delphi configuration tests pass.

The first full Iris run reached 309 passing tests. Seven runtime-bundle tests rejected the intentionally uncommitted
`rl_config_translation.py`, as required by their source-identity contract. The full suite must run again after the branch
checkpoint is committed.

## Future work

- [ ] None.
