# Debugging log for Ray object-store pressure

## Initial status

An E9 policy forward at 24k context serialized padded training shards across 64 policy ranks. Dispatch
took 561--588 seconds and crossed the 480-second ProcessGroupNCCL heartbeat deadline. Ray spilled about 2.05
TiB through a CoreWeave R2 bucket despite the run using little host memory. At 16k context the same dispatch took
261--289 seconds and completed.

The current mesh dispatch shares an explicit `ObjectRef` only when a batch contains `rollout_routed_experts`.
Ordinary dense batches are passed by value to every actor. Ray documents that repeated large by-value arguments
create one object-store copy per task. The Iris controller also selects the `smart_open` R2 spill backend by
default and sets `min_spilling_size=0`.

## Hypothesis 1: per-rank by-value dispatch is the dominant data multiplier

A large non-router-replay batch sent to eight actors across four data-parallel groups should serialize four
times: once per distinct data chunk. The current implementation serializes it once per actor.

## Results

Partly refuted as the explanation for E9. The 64 FSDP ranks consume distinct DP shards, so sharing an ObjectRef
cannot collapse those 64 values into one. Existing router-replay dispatch already shares chunks when several
model-parallel ranks consume the same DP shard. Generalizing that transport does not address this incident and
is not part of this change.

The discriminating serialization probe found the larger multiplier. A 64-row, 8 MiB parent tensor produced
128 KiB one-row chunks whose views still referenced the parent's full 8 MiB storage. `torch.save` serialized
8,390,185 bytes for each 128 KiB view. Pickling `TensorBatch` then serialized both the inherited dictionary and
the custom `__getstate__` payload, producing 16,779,367 bytes per chunk. After compacting sliced storage and
making the custom reducer authoritative, the same chunk serializes to 132,814 bytes and round-trips exactly.
Across all 64 chunks, Ray cloudpickle now writes 8,500,608 bytes for an 8,388,608-byte logical tensor, a 1.013x
ratio.

This establishes two independent amplification bugs: full-parent storage serialized for every DP shard, and a
duplicate inherited-dictionary payload. The regression test serializes every shard and applies one aggregate
budget, so either multiplier makes it fail.

## Hypothesis 2: remote spill is an unsafe default for ephemeral training arguments

Training-step arguments are reconstructible, latency-sensitive data. They should spill to a launcher-owned local
scratch directory by default. Durable R2 spill remains available only through an explicit operator opt-in.

## Changes to make

Add controller tests requiring local default spill flags and explicit remote-spill opt-in. Keep the rendezvous,
logs, and termination artifacts on durable object storage; this change applies only to Ray's ephemeral object
store.

## Results

The controller tests failed because remote R2 was enabled when the opt-in variable was absent and local mode
emitted no explicit spill directory. The implementation now defaults to launcher-owned local scratch and keeps
R2 behind `OT_AGENT_RAY_SPILL_TO_R2=1`.

## Future work

- Measure tensor bytes, padding bytes, primary object-store residency, and cumulative spill/restore bytes per
  training phase in a production-sized run.
- Design a ragged wire representation for dense padded batches, with reconstruction at the worker boundary.
- Set a cluster-wide spill-byte circuit breaker from measured healthy-run distributions. Do not guess a default
  that could kill legitimate long-context work.
