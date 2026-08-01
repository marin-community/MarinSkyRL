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
and model worker collect together on macOS: all 12 accelerator cases skip because CUDA is unavailable. The
four-GPU matrix remains the acceptance gate before merge.

The standalone model contract in #270 passed EP2/FSDP2 on four GH200 GPUs before this matrix was added. That
result recorded 24 EP operations and 21 FSDP operations with reentrant checkpointing and concentrated replay
routing. It does not validate the other five matrix cases or the four-node EP4/FSDP4 Jupiter geometry.
