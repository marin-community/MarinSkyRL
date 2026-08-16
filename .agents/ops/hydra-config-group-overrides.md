# Debugging log for packaged Hydra config-group overrides

Allow Iris RL configs to override fields already declared by the packaged Terminal-Bench Hydra group.

## Initial status

`build_skyrl_hydra_args()` emits every `terminal_bench_config` leaf with Hydra's single-plus append prefix. The packaged
`terminal_bench` group now declares `prm` and `trace_upload`, so composition rejects ordinary experiment overrides.

## Hypothesis 1

The add-or-override `++` prefix should compose whether a leaf exists in the packaged group or is introduced by an
experiment config.

## Changes to make

- Reproduce the failure by composing the real TaskTrove launcher arguments with the packaged trainer config.
- Verify that every packaged `prm` and `trace_upload` field retains the experiment value after composition.
- Emit Terminal-Bench leaves with `++` and remove the stale single-plus comment.

## Results

The real TaskTrove arguments reproduced Hydra's `Could not append to config` failure at
`terminal_bench_config.trace_upload.enabled`. With `++`, the complete launcher argument list composes against
`ppo_base_config`, and the packaged `prm` and `trace_upload` fields retain the experiment values.

## Future work

- [ ] None.
