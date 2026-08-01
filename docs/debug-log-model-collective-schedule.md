# Debugging log for model collective schedules

Build a small regression contract for the remaining rank-divergence question after the worker-level EP replication and co-arrival mechanisms were verified.

## Initial status

The existing four-GPU expert-parallel test exercises TorchTitan EP and forced routing without FSDP2 or activation checkpoint recomputation. The full RL tests compose those mechanisms only with multi-billion-parameter checkpoints, and make training completion the oracle. Neither records the EP and FSDP schedules at model-layer boundaries, so a later mismatch is observed only as a timeout.

## Hypothesis 1

A tiny grouped MoE can preserve the production process-group topology and ordering constraints without the production model size. With distinct data and replay targets across FSDP replicas, every EP group must still issue the same dispatch schedule and every FSDP group must still issue the same unshard/reduction schedule.

## Changes to make

- Compose the production TorchTitan EP hooks, FSDP2 wrapping, grouped router replay, and reentrant activation checkpointing in a three-layer model.
- Record completed NCCL operation types and sequence numbers from each rank's EP and FSDP process groups without inserting collectives.
- Snapshot both process-group sequence counters at every original-forward and reverse-order recompute layer boundary.
- Compare schedules only among members of the same process group and report the first missing operation or shifted boundary.

## Results

The pure schedule comparison is covered on CPU, including independent EP/FSDP grouping, a missing EP combine, and layer-boundary sequence drift. All 72 distributed CPU tests pass, with one existing platform skip. The GPU module imports and collects with the locked EP dependencies, then skips because CUDA is unavailable. The four-rank and optional twelve-rank NCCL contract still requires a GPU run before merge.
