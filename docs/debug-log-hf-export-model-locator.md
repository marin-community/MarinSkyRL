# Debugging log for HF export model locators

Standalone Hugging Face export requests must preserve a model reference that a later Iris gang can materialize.

## Reported failure

Four export requests recorded only the training container path `/tmp/snowball-grug-67b-a2b-sft-s2-thinking-step630`. The later export gangs did not have that directory. The launcher fabricated an S3 warm-cache key from the local path, then retried Hugging Face prestaging six times with the same invalid repo ID.

## Hypothesis 1

The launcher's durable `model_source_uri` and `model_source_identity` stop at task bootstrap. `training_driver` writes only the task-local path into the trainer config, so `HFExportRequest` cannot distinguish a Hub repo ID from an ephemeral materialization path.

## Experiment

Exercise request creation with a task-local model plus an object-store source, request validation with a task-local model alone, and offline task-command construction with the same locator. These tests cover provenance propagation, submission-time rejection, and warm-cache derivation.

## Results

The trainer request omitted both durable fields, local-only requests passed validation, and the offline task command contained `--prestage-model /tmp/materialized-model` plus the fabricated `models/--tmp--materialized-model` warm prefix. Threading the typed locator through `training_driver`, validating it at request construction and consumption, and limiting Hugging Face prestaging to repository IDs made all three cases pass. The export command now rematerializes object-store models before loading the checkpoint.
