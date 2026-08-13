# Debugging log for Megatron offload checkpointing

Make colocated Megatron checkpoints durable when the policy is offloaded for inference.

## Initial status

Periodic checkpoint callbacks run after the colocated policy model and optimizer have been offloaded. Megatron's
distributed checkpoint serializer then reads parameter views whose GPU storage has size zero.

## Hypothesis 1

Checkpointing must own a temporary residency transition: sleep the colocated inference engines, backload the policy
model and optimizer, save, then restore rollout residency in a `finally` block.

## Changes to make

Add a trainer checkpoint boundary that manages colocated residency and use it for periodic and final saves. Add
regression coverage for successful and failed saves, failed timer reporting, and an empty latest-resume prefix.

## Results

The regression tests failed before the fix: the trainer had no residency-aware save boundary, the timer emitted
`Finished` while propagating an exception, and an empty latest checkpoint prefix logged at INFO. The trainer now
backloads both model and optimizer around periodic colocated saves and restores inference residency in `finally`.
Because the inference sleep releases both weights and KV cache, restoration also re-synchronizes policy weights before
waking the KV cache.

The focused trainer regressions and the complete timer test file pass on the CPU profile. The full `test_trainer.py`
file still has the existing `test_ppo_train_batch_calculations` fixture failure because its inline algorithm config
omits the required `batch_invariant` key; this change does not touch that worker initialization path.

## Future work

- [ ] Run an on-GPU Megatron checkpoint smoke test when an idle allocation is available.
