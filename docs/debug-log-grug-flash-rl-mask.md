# Debugging log for Grug FlashAttention RL masks

Allow Grug FlashAttention to consume the RL trainer's left-padded prompt plus right-padded response layout without weakening its mask contract.

## Initial status

Issue #254 reports that the first policy forward fails because an RL-shaped mask such as `[0, 1, 1, 0]` has two transitions. The guard permits at most one, even though the valid tokens form one contiguous span.

## Hypothesis 1

The mask validator is narrower than the FlashAttention varlen implementation. `flash_attn.bert_padding.unpad_input` selects every valid token and records each row's valid length, so an interior contiguous span needs no new unpadding mechanism.

## Changes to make

Add a CPU model-forward regression test that reaches the flash backend with an RL-shaped mask, and add GPU eager/FlashAttention output and gradient parity coverage for the same layout. Run the CPU test before changing the validator to demonstrate the regression.

## Results

After installing the package's development dependencies, the focused CPU regression test failed in `_validate_flash_attention_mask`: `[0, 1, 1, 0]` produced two transitions and raised the reported `RuntimeError` before attention ran. This confirms the guard as the immediate cause.

## Hypothesis 2

Counting valid-span starts expresses the intended contract directly: a nonempty row is supported exactly when it has one false-to-true boundary, including a valid token in column zero as a boundary. Dense, left-padded, right-padded, and interior-span masks each have one start; discontiguous masks have more than one.

## Changes to make

Replace the transition limit with the span-start invariant. Retain the asynchronous tensor assertion to avoid adding a device-to-host synchronization to every training forward. Update the rejection test's message while preserving its discontiguous-mask coverage.

## Results

The focused FlashAttention tests pass (3 passed), including the RL-shaped model forward and the existing empty/discontiguous rejection cases. The complete Grug CPU model test file also passes (16 passed). A direct invariant check covers dense, left-padded, right-padded, interior-span, and discontiguous masks.

The H100 parity test now includes the interior-span layout and will exercise the real FlashAttention varlen kernel. It is not runnable on the local macOS host.

The complete `skyrl-train` CPU suite finished with 880 passed, 19 skipped, and two unrelated failures: `test_generator_output_concatenation` has a stale expected field inventory, and `test_all_defaults_is_structurally_identical_to_pre_ep` has a stale golden config. This branch does not change either failing test or its implementation inputs.

## Future work

- [ ] Run `tests/gpu/test_grug_flash_attention.py` on an H100 gate.
