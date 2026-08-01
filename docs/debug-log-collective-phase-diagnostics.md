# Debugging log for collective phase diagnostics

Make a future EP/FSDP schedule divergence reconstructable from durable worker
logs without adding synchronization to the training path.

## Initial status

The existing optional diagnostic recorded only the default process-group
sequence number. The observed failure split ranks between EP and FSDP subgroup
collectives, so the recorded counter could not identify either subgroup's first
disagreement. Its first-MoE guard also used thread-local state, while an FSDP
inference forward crosses from an async actor task into `asyncio.to_thread`.
Policy training did not establish a new diagnostic region per microbatch or
separate the original forward from backward recomputation.

## Hypothesis 1

Reading the existing sequence number from the world group and every named
device-mesh group at a few policy phase boundaries is enough to distinguish a
phase mismatch, a subgroup count mismatch, and a rank that never reaches the
next boundary. A context variable containing mutable per-region state preserves
the event order across `asyncio.to_thread` without issuing a collective.

## Changes to make

- Emit structured, event-ordered records for world and named mesh groups.
- Start distinct regions for inference forwards and policy-training
  microbatches, including global and local step metadata.
- Reset the first-MoE marker for the original forward and backward recompute.
- Parse the records and report the earliest missing rank, phase mismatch, or
  process-group sequence mismatch.
- Preserve a zero-touch disabled path and warn without interrupting training if
  an enabled capture cannot inspect its mesh.

## Results

The red tests showed that the original module had no subgroup record, structured
parser, cross-thread region, or first-divergence comparison. Seven focused CPU
contracts now pass for capture, serialization, cross-thread ordering, phase
resets, disabled behavior, capture failure, matching schedules, subgroup
divergence, and missing-rank detection. Adjacent policy-worker and launcher
tests remain green. The complete trainer CPU suite passes with 908 tests and 19
skips.

## Future work

- [ ] Validate the emitted sequence records during an on-demand multi-node
  EP/FSDP training step before enabling the diagnostic in a long run.
