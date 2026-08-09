# Debugging log for standalone HF export

Replace the trainer-resume export path with a policy-only checkpoint conversion utility.

## Failure evidence

The former export path launched the configured RL entrypoint and selected its conversion-only trainer only after
constructing the RL experiment. Four 64-GPU policy exports consequently requested an additional 16 rollout GPUs
and timed out before loading their checkpoints.

## Hypothesis 1

The export specialization occurs below the resource-allocation boundary. Selecting a dedicated entrypoint before
the RL experiment is constructed will prevent rollout, dataset, tracking, reference, and critic initialization.

## Design

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

The complete launcher and trainer CPU gate passes, including the runtime-bundle identity checks
that ensure Iris executes the committed checkout rather than stale installed package bytes.

## Accelerator validation

An on-demand accelerator run should validate one small FSDP2 export and one Megatron export at each
checkpoint's saved model-parallel geometry before either path is used for a production model.

## Review dispositions

The publisher and worker protocols are intentional test seams: conversion must prove that a failed or
incomplete artifact is never published without constructing Ray actors or making Hub calls. The standalone
entrypoint continues to use the production strategy converters so checkpoint and Hugging Face formats have
one owner; only their training-state initialization is bypassed.

The Iris launcher keeps export decisions in its shared job builder because both training and conversion must
produce the same task protocol, runtime identity, and source bundle. A single job-kind predicate owns the
distinction; the guarded sites suppress training-only defaults or resources rather than implementing separate
conversion behavior.
