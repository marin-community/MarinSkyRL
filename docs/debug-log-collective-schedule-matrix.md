# Debugging log for the collective schedule matrix

Extend the compact EP/FSDP model contract with controlled perturbations that can expose deterministic rank
divergence without requiring a production-size model.

## Initial status

The model-level contract in PR #270 composes TorchTitan EP, FSDP2, grouped router replay, and reentrant activation
checkpointing in one configuration. It can detect a per-process-group schedule mismatch, but a pass does not
separate the effects of live routing, replay, concentrated expert selection, activation recomputation, or
rank-arrival skew.

## Hypothesis 1

Fresh four-rank gangs can isolate those axes while retaining the real EP and FSDP model hooks. A healthy
implementation should produce matching schedules within each process group for live and replayed routing, with
and without reentrant checkpointing, under spread and concentrated expert selections, and after one rank arrives
late at a model-layer boundary.

## Changes to make

- Parameterize the tiny model contract by checkpoint, routing, and controlled-delay modes.
- Launch every matrix case in its own torchrun gang so process groups and NCCL completion hooks cannot leak
  between cases.
- Keep the matrix outside default pytest discovery and bound each gang plus cleanup independently.
- Document the exact opt-in command and evidence required from a GPU run.

## Results

The shared schedule comparison passes 3 CPU contracts. The matrix controller, existing fault-injection suite,
and model worker collect together on macOS: all 12 accelerator cases skip because CUDA is unavailable.

All six matrix cases passed on four GH200 GPUs at `b833cfae` in 121.96 seconds. Each case used a fresh four-rank
gang, and bounded cleanup left no process group behind. The run used the policy image's installed pytest and
CUDA stack.

The previously documented isolated `uv` command could not run on the air-gapped compute node because the lock
contains a direct GitHub URL for `harbor-config`. Resolving it on a networked login node selected a CUDA 12.9
Torch build instead of the image's CUDA 13.0 aarch64 build. The GPU documentation now uses the installed image
environment; isolated resolution is not evidence about the built runtime.

The standalone model contract in #270 passed EP2/FSDP2 on four GH200 GPUs before this matrix was added. That
result recorded 24 EP operations and 21 FSDP operations with reentrant checkpointing and concentrated replay
routing. It does not validate the other five matrix cases or the four-node EP4/FSDP4 Jupiter geometry.
