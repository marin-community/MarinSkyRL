# Debugging log for nightly issue 408

Diagnose and fix the `gsm8k-h100` failure in scheduled Actions run 32121896630.

## Initial status

The `grug-vllm-gb200` job passed. Iris scheduled `marinskyrl-nightly-gsm8k-h100-32121896630-1`, installed the frozen runtime, and started the trainer. The reference FSDP worker then failed during model setup with `ConfigAttributeError`: `trainer.ref.fsdp_config.expert_loader_chunk_rows` was absent.

## Hypothesis 1

PR #407 moved the expert-loader chunk size from an environment variable to typed Hydra configuration, but added the new field only to `trainer.policy.fsdp_config`. `FSDPStrategy._fsdp_init_model` reads the field for policy, reference, and critic models, so the default reference and critic configurations are invalid.

## Changes to make

First extend the shared FSDP default regression test to require `expert_loader_chunk_rows` for all three model roles and confirm that it fails. Then add the typed default to the reference and critic FSDP configurations and validate every role's value.

## Results

The shared-default test failed on unmodified `main` with `trainer.ref.fsdp_config missing expert_loader_chunk_rows`, matching the nightly traceback. The reference and critic defaults now interpolate the authoritative policy value. The completed configuration, placement, and nightly subset passed 42 tests. The required changed-file lint passed.

The full CPU CI command collected 1,535 tests and reached the later trainer tests without an assertion failure, then the constrained session was killed with exit 137 after repeated local Ray instances. A fresh run including `test_trainer.py` later inherited an unavailable Ray GCS address and was interrupted. These resource failures occurred outside the changed configuration tests.

## Future work

- [x] Run the narrow configuration regression test.
- [x] Run the relevant CPU configuration tests and repository checks.
