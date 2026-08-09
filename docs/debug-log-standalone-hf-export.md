# Debugging log for standalone HF export

Replace the trainer-resume export path with a policy-only checkpoint conversion utility.

## Initial status

An export request launches the configured RL entrypoint with `trainer.hf_export_execution=true`. The entrypoint
still allocates rollout placement groups, inference engines, a generator, tracking, and any configured reference
or critic models before selecting `CheckpointExportTrainer`. Four 64-GPU policy exports therefore requested an
additional 16 rollout GPUs and timed out before loading their checkpoints.

## Hypothesis 1

The export specialization occurs below the resource-allocation boundary. Selecting a dedicated entrypoint before
the RL experiment is constructed will prevent rollout, dataset, tracking, reference, and critic initialization.

## Changes to make

- Route export jobs to a dedicated Hydra entrypoint.
- Give that entrypoint a policy-only worker-group controller with an explicit checkpoint/export contract.
- Initialize model structure without optimizer, scheduler, profiler, or weight-sync state.
- Load model tensors without optimizer or scheduler state and invoke the existing FSDP2 or Megatron HF converter.
- Add export runtime profiles that omit vLLM, Harbor, Daytona, and training telemetry dependencies.

## Results

The focused CPU contracts pass. Export commands select the dedicated entrypoint and policy-only
node count. The in-container driver parses only the policy configuration and emits no data,
generator, environment, or synthesized training-path arguments. The Ray adapter restores model
tensors without optimizer, scheduler, RNG, or callback state. The FSDP checkpoint test confirms
that conversion stages only the rank's model shard. Export runtime profiles resolve without the
vLLM, Harbor, Daytona, and telemetry extras used by training jobs.

The full launcher and trainer CPU run reached 1,282 passing tests. Five runtime-bundle tests were
intentionally blocked by the repository's dirty-bundle guard and will be rerun after commit; the
only other failure was a checked-in configuration baseline updated by this branch and is green
after adding the dedicated export section.

## Future work

- [ ] Validate a small FSDP2 export on an accelerator after the CPU contract and Iris launcher tests pass.
- [ ] Validate a Megatron export at its saved model-parallel geometry.
