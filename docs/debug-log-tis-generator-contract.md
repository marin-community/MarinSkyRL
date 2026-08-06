# Debug log: TIS on non-batched generation

## Reported failure

With `trainer.algorithm.use_tis=true`, config validation enables rollout
logprobs. `SkyRLGymGenerator` then rejects the otherwise valid default
`generator.batched=false` configuration during construction.

## Evidence

- `validate_cfg` enables `generator.sampling_params.logprobs` for TIS but does
  not require a particular generation strategy.
- `SkyRLGymGenerator._validate_cfg` rejects logprobs unless `batched=true`.
- The non-batched agent loop already receives `response_logprobs` from the
  inference-engine client but does not read them.
- Fully asynchronous training requires `generator.batched=false`, so forcing
  batching would make TIS incompatible with that trainer.
- Commit `7cd23ea1` previously supported non-batched TIS by accumulating each
  turn's generated-token logprobs and zero-filling masked context tokens. The
  support was lost when an older generator snapshot was restored later.
- The existing trainer callback API only exposes lifecycle events and returns
  `TrainerControl`; it has no rollout transformation event or output return
  contract. `GeneratorOutput.rollout_logprobs` is already the shared data
  contract consumed by synchronous and fully asynchronous trainers.

## Hypothesis

The crash is an accidental capability regression, not an inherent TIS/batching
constraint. Capturing logprobs in the token-in/token-out agent loop and aligning
masked context positions will make TIS independent of batching without changing
trainer-specific code.

## Test plan

1. A TIS-valid config may retain `generator.batched=false` while validation
   enables logprob capture.
2. Non-batched, single-turn generation returns engine logprobs aligned one-for-
   one with response tokens.
3. Non-batched, multi-turn generation preserves per-turn logprobs and inserts
   neutral placeholders only at loss-masked observation/prompt positions.
4. If an environment rewrites generated text, discard the now-invalid rollout
   logprobs rather than attaching them to different token IDs.

## Results

- Before the implementation, the config contract test passed but all three
  non-batched behavior tests failed at generator construction with
  ``ValueError: sampling_params.logprobs should be None if batched is False``.
- The implementation removes that strategy gate and accumulates exact engine
  logprobs through both token-in/token-out chat layouts. Loss-masked context
  positions receive neutral placeholders so all three output channels remain
  aligned.
- Text postprocessing invalidates token identity and now drops the whole
  rollout-logprob channel with a warning.
- The focused generator suite passes: 26 passed.
- The repository CPU gate passes: 1,251 collected, 9 skipped.
