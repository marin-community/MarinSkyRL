# MoE dispatch discriminator

Build a standalone EP4/FSDP4 contract that records the first rank-local loss of progress inside a plain MoE
forward without changing production training behavior.

## Initial status

Natural Jupiter failures identify a first nonparticipant after the first MoE boundary. Peers subsequently time
out in EP `ALLTOALL_BASE` and FSDP `_ALLGATHER_BASE`, but the existing boundary is too coarse to distinguish a
rank that never enqueued token dispatch from one whose CUDA stream stalled after enqueue.

The compact model matrix passes live routing, replay, checkpointing, concentrated routes, and bounded rank
delay on one node. It does not exercise the production EP4/FSDP4 topology or emit dispatch-stage records.

The backend-numerics escalation is partly stale. T2 and T3 were added in PR #373, but T3 labels an MoE case as
grouped while constructing it with `use_grouped_mm=False`; it therefore does not execute the grouped kernel.

## Hypothesis 1

A test-only wrapper around TorchTitan's production `ExpertParallel._token_dispatch` can expose routing/count
completion, both all-to-all enqueue boundaries, and CUDA completion without copying the dispatch algorithm.
A multi-node controller can validate that every rank crosses the same ordered stages for multiple ragged
microbatches and preserve the structured records for comparison with the next natural timeout.

## Changes to make

- Add a structured dispatch-stage protocol and CPU validation contracts.
- Run a tiny grouped-MoE model through the production TorchTitan EP and FSDP2 wrappers on EP4/FSDP4.
- Exercise multiple plain forwards with different sequence lengths across FSDP replicas and identical inputs
  within each EP replication group.
- Persist raw JSONL records and a validation summary outside the disposable process-gang directory.
- Add an honest BF16 grouped-kernel case to T3 and retain the FP32 for-loop oracle separately.

## Results

Pending local and Jupiter execution.
