# Debugging log for shard-timeout output schema

Prevent terminal shard timeouts from crashing or corrupting fully asynchronous batch concatenation.

## Initial status

The synthetic output introduced by PR #379 omits `unshaped_rewards`. A normal output followed by a timeout
output raises `KeyError` during concatenation. Reversing the order silently removes the raw outcome channel.

## Hypothesis 1

All-failed terminal-bench outputs must carry the same raw outcome channel as normal outputs. Concatenation also
needs to tolerate an incomplete group so a mixed batch cannot depend on which group finishes first.

## Changes to make

Reproduce both group orders through `concatenate_generator_outputs`. Add zero-valued `unshaped_rewards` to the
all-failed output and preserve raw outcomes across mixed complete and incomplete groups during concatenation.

## Results

Before the fix, normal-first concatenation raised `KeyError: 'unshaped_rewards'`; timeout-first concatenation
returned no raw outcome channel. After the fix, both orders retain one raw outcome per trajectory. The focused
generator, reward-shaping, timeout, and trainer utility suite passes with 126 tests.

## Future work

- [ ] None identified.
