# Debugging log for opt-in Hugging Face export

Prevent periodic Hugging Face export requests when training has no explicit Hub repository.

## Initial status

The default trainer configuration leaves `trainer.hf_hub_repo_id` null but sets `trainer.hf_save_interval` from `trainer.ckpt_interval`. The legacy callback builder therefore installs `HFModelSaveCallback` for ordinary training. At an interval boundary, the callback asks `RayPPOTrainer` to queue an export request; task-local model paths without durable source metadata then raise `ModelLocatorError` and terminate training.

## Hypothesis 1

Legacy interval-based export must be considered disabled unless both `hf_save_interval` is positive and `hf_hub_repo_id` is set. Configuration validation must use the same enablement condition.

## Changes to make

Add behavior-level callback tests for absent and explicit repository IDs, plus a validation regression for an irrelevant export interval when no repository is configured.

## Results

Before the fix, the interval-only callback regression set `should_save_hf_model=True` at step 5, and configuration validation rejected a 5-step export interval against a 3-step checkpoint interval even though `hf_hub_repo_id` was null.

Added one shared legacy-export predicate that requires a non-empty repository ID and a positive interval. Default callback construction and interval-alignment validation now use the same predicate. Explicit callback lists remain explicit and retain their existing behavior.

The focused HF export suite passed with 21 tests, the adjacent trainer suite passed with 17 tests, and the complete trainer CPU suite passed with 1,118 tests and 20 skips.

## Future work

- [ ] None.
