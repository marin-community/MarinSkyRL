# Debugging log for EP replication and co-arrival contracts

Establish durable regression coverage for the historical EP-aware data replication and policy-entry co-arrival fixes before investigating the remaining mid-training collective divergence.

## Initial status

The production code contains the topology and co-arrival fixes, but the CPU suite only covers adjacent behavior. Dispatch tests construct `MeshRank` values manually, while EP mesh tests do not exercise the worker's data-replication rank derivation. No test requires the fully asynchronous trainer to finish a policy drain immediately before forward dispatch, or requires the worker's entry barriers and forward body to run without blocking the actor event loop.

## Hypothesis 1

The historical mechanisms work as described, but can regress without failing current CI.

## Changes to make

Add behavior-level CPU contracts through the production entry points:

- derive every worker's dispatch rank for the incident EP/CP topologies and verify one shared data shard per replication group;
- require the fully asynchronous training step to finish its drain before policy forward begins;
- require the worker drain barrier and FSDP inference forward to move blocking work off the actor event-loop thread;
- require decentralized router-replay training to synchronize CUDA and then enter the world barrier before touching the training loop.

## Results

All six contracts pass against current `main`. The topology cases cover both the earlier 32-rank CP2/EP8 geometry and the 12-rank CP1/EP4 geometry from the recent incident. The ordering cases confirm that the fully asynchronous trainer awaits the drain before forward, the worker drain executes CUDA synchronization before the world barrier on a background thread, the FSDP inference body leaves the actor event-loop thread, and decentralized router-replay training reaches its entry barrier before later training state.

The full CPU suite completed with 897 passes, 19 skips, and two failures. Both failures reproduce unchanged on an untouched `main`: `test_generator_output_concatenation` has a stale expected field list, and `test_all_defaults_is_structurally_identical_to_pre_ep` has a stale golden configuration. The new contract file passes all six tests, and the adjacent dispatch, EP mesh, CP mesh, and trainer tests pass all 24 tests.

This confirms the mechanisms are present and working at their Python orchestration boundaries. It does not exercise NCCL timing or prove that later model-internal collective schedules remain identical; those are the next two stages.

## Future work

- [ ] Compare real model collective schedules per process group.
- [ ] Exercise checkpointing, router replay, routing skew, and controlled delays in the opt-in GPU suite.
- [ ] Persist first-divergence diagnostics from production workers.
