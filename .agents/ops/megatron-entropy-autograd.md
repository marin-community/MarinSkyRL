# Debugging log for Megatron entropy autograd

Fix the deterministic first-step failure when Megatron training combines policy log-probability and entropy losses.

## Initial status

The X16 entropy arm failed all seven retries in its first optimizer step. `_VocabParallelLogProbs` and
`_VocabParallelEntropy` save the same logits tensor for backward, while entropy backward subtracts from and restores that
tensor in place. The value is restored, but the version counter changes and log-probability backward rejects the saved
tensor.

## Hypothesis 1

Combining vocab-parallel entropy and log-probability losses over the same logits reproduces the saved-tensor version error,
even with a one-rank CPU process group.

## Changes to make

Add a numerical regression that computes both losses from one logits tensor and compares their combined gradient with an
independent PyTorch reference.

## Results

Confirmed. The regression fails in `ChunkedDistributedLogprob.backward` because the shared logits tensor is at version 2
instead of the saved version 0.

## Hypothesis 2

Building the entropy gradient from an out-of-place subtraction, then updating only that new gradient buffer in place, will
preserve the input logits and match the independent softmax/log-softmax gradient.

## Changes to make

Replace entropy backward's subtract/restore sequence on `vocab_parallel_logits` with one out-of-place subtraction and
in-place arithmetic confined to the resulting gradient buffer.

## Results

Confirmed. The combined regression passes and its logits gradient matches the independent PyTorch reference at
`atol=1e-6, rtol=1e-6`. The distributed CPU suite passes with 98 tests passed and one skipped.

## Future work

- [ ] Consider a bounded-memory entropy implementation if the logits-sized backward temporary is material in production.
