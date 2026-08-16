# MoE routing recompute desynchronization

## Symptom

FSDP2 expert-parallel training can fail during activation-checkpoint backward because the recomputed routed-token
count differs from the original forward by one token. The resulting expert all-to-all split mismatch surfaces as a
checkpoint determinism error or NCCL abort.

## Reproduction

- Use non-reentrant activation checkpointing around the grouped MoE.
- Make two router logits nearly tied and reverse their order on recomputation.
- Observe that backward sends gradients through a different expert than the original forward selected.

## Hypothesis log

1. The grouped MoE recomputes `torch.topk` during backward and has no forward-local route replay. **Confirmed** from
   `MoE.forward`: the `routed_experts` hook is populated only by rollout-time R3, not activation checkpointing.
2. A checkpoint-scoped record/replay context can retain the forward indices and reuse them during recomputation without
   enabling rollout-time router replay or changing the selected experts of the training forward. **Confirmed.**

## Results

- Before the MoE integration, the regression fails during backward with `CheckpointError`: saved expert input shapes
  `[2, 2]` become `[0, 2]` on recomputation and vice versa because every token switches experts.
- With checkpoint-local route replay, the regression passes and only the forward-selected expert receives a gradient.

## Review responses

- Kept the test's narrow `torchtitan` decorator stub instead of skipping when `torchtitan` is absent. The CPU dependency
  closure deliberately omits `torchtitan`; skipping would remove this regression from the PR gate. The stub is used only
  to import the EP=1 for-loop path and is removed from `sys.modules` immediately afterward.
