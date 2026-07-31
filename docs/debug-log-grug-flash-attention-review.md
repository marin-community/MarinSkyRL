# Debugging log for Grug FlashAttention review

Review the fused Grug attention path against its eager reference, supported mask contract, production loader wiring, and numerical test policy.

## Initial status

The fused implementation passed its original CPU and H100 checks. It supported full-causal and sliding-window attention with left-padded batches.

## Mask-contract hypothesis

The varlen path compacted arbitrary boolean masks even though only dense, left-padded, and right-padded rows are supported. Holey masks changed sliding-window semantics, and empty rows reached the CUDA kernel.

## Changes to make

Validate the mask once at the model boundary with asynchronous tensor assertions. Reject empty rows and rows with more than one validity transition.

## Results

The public model forward now rejects empty and holey rows before entering FlashAttention. Dense and one-sided padded rows remain supported without a per-layer host synchronization.

## Loader and numerical-gate hypothesis

The RL-cycle test enabled FlashAttention in configuration but did not observe the backend loaded by policy workers. The kernel parity test printed maximum and mean errors without gating them.

## Changes to make

Read the loaded attention backend from every policy rank during the RL cycle. Gate maximum and mean output and QKV-gradient errors using ceilings above the measured H100 errors but tight enough to catch meaningful drift, and cover right-padded parity.

## Results

CPU coverage observes invalid-mask rejection and production backend selection is part of the H100 RL-cycle contract. The H100 parity test now gates both pointwise and aggregate errors for left padding, right padding, and full-causal attention.

## Future work

- [ ] Run the updated H100 parity, memory, and four-GPU RL-cycle checks.
