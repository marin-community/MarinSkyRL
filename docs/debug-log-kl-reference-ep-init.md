# Debug log: KL reference model FSDP2 expert-parallel initialization

## Goal

Make the FSDP2 KL reference model use the configured grouped-MoE representation before expert sharding, and
prove that a real reference forward produces finite log probabilities with EP enabled.

## Initial status

- The Qwen3-Coder policy model initializes with `expert_model_parallel_size=4` and
  `moe_grouped_gemm=true`.
- The otherwise identical reference model reaches `apply_ep` with no supported grouped experts, so
  `apply_ep` returns zero and initialization fails.
- The no-KL control does not instantiate a reference model and trains normally.

## Hypothesis 1: the reference worker drops the MoE construction settings

The policy worker passes `moe_router_replay`, `moe_grouped_gemm`, and `use_grouped_mm` from its FSDP config
to `HFModelWrapper`. The reference worker passes none of them. Their `FSDPStrategy` instances therefore see
the same EP topology but different model structures: the policy has `GroupedMoEShim` modules and the
reference retains the eager Hugging Face expert blocks.

### Evidence

- `FSDPPolicyWorkerBase.init_model` forwards all three settings.
- `FSDPRefWorkerBase.init_model` forwards none of them, despite reading the reference FSDP config for mesh
  construction.
- `apply_ep` deliberately recognizes grouped expert holders only; retaining the assertion is necessary to
  prevent silently unsharded expert parameters.
- Git history shows the reference constructor predates the grouped-MoE/EP feature and was not updated when
  those policy arguments were added.

### Test plan

Add an opt-in four-GPU regression that creates a tiny local Qwen3 MoE checkpoint, initializes a real FSDP2
reference actor group with EP=2 and grouped MoE enabled, and runs the production reference forward. The test
must fail at the reported `apply_ep` assertion before the fix and pass only if the returned reference log
probabilities are finite and have the expected shape.

## Changes and results

- Added a four-GPU, EP=2 x FSDP=2 reference-worker regression using a tiny local Qwen3 MoE checkpoint.
- Pre-fix Jupiter job 1282483 reproduced the production failure at `fsdp_strategy.py:423`: all ranks loaded
  eager HF experts and `apply_ep` returned zero.
- Centralized the three model-structure settings (`moe_router_replay`, `moe_grouped_gemm`, and
  `use_grouped_mm`) in one role-config mapping used by both policy and reference constructors.
- Applied the policy path's Grug/EP compatibility validation to reference construction as well.
- Post-fix Jupiter job 1282519 completed successfully on the final source: all four ranks grouped and
  EP-sharded the reference model, executed the production reference forward, returned finite log
  probabilities of the expected shape, and Ray/Slurm exited cleanly (`1 passed`, job exit 0).

## Future work

- The opt-in test uses a tiny Qwen3 MoE rather than the 30B production checkpoint so it can isolate model
  construction, expert sharding, and reference forward without coupling the regression to model downloads or
  the TaskTrove harness.
