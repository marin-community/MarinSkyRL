# Debugging log for Megatron offload checkpointing

Make colocated Megatron checkpoints durable when the policy is offloaded for inference.

## Initial status

Periodic checkpoint callbacks run after the colocated policy model and optimizer have been offloaded. Megatron's
distributed checkpoint serializer then reads parameter views whose GPU storage has size zero.

## Hypothesis 1

Checkpointing must own a temporary residency transition: sleep the colocated inference engines, backload the policy
model and optimizer, save, then restore rollout residency in a `finally` block.

## Changes to make

Add a trainer checkpoint boundary that manages colocated residency for periodic saves. Final saves already run after
the inference engines sleep and the policy is backloaded. Add regression coverage for successful and failed saves
and failed timer reporting. Warn when `resume_mode=latest` finds no checkpoint marker.

## Results

The regression tests failed before the fix: the trainer had no residency-aware save boundary, the timer emitted
`Finished` while propagating an exception, and an empty latest checkpoint prefix logged at INFO. The trainer now
backloads both model and optimizer around periodic colocated saves and restores inference residency in `finally`.
Because the inference sleep releases both weights and KV cache, restoration also re-synchronizes policy weights before
waking the KV cache.

## Future work

- [ ] Run an on-GPU Megatron checkpoint smoke test when an idle allocation is available.
