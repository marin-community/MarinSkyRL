# Debugging log for overlong initial prompts

Keep a templated prompt that already exceeds the generation input budget from
raising during agent-loop cleanup.

## Initial status

`agent_loop` checks the templated prompt length at the top of its turn loop. An
overlong initial prompt exits before generation, but the token-in/token-out
cleanup still reads `response_end_idx`, which is assigned only after a generated
turn. The retokenizing cleanup similarly depends on `new_obs` and a recorded
step reward that do not exist on a zero-turn exit.

## Hypothesis 1

An explicit zero-turn result at the initial length boundary can preserve the
normal `length` stop reason without entering cleanup paths that require a
generated turn. Empty token-level rewards must also remain valid when rollout
metrics aggregate the batch.

## Changes to make

- Add a regression covering both agent-loop tokenization modes.
- Return an empty, zero-valued rollout before inference when the templated
  initial prompt exceeds `max_input_length`.
- Treat an empty token-level reward as zero in rollout reward metrics.

## Results

The regression failed on the original code with `UnboundLocalError` for
`response_end_idx` in token-in/token-out mode and `new_obs` in retokenizing
mode. After the explicit zero-turn return, the focused regression passed in
both modes, the generator module passed 19 tests, and the complete trainer CPU
suite passed 907 tests with 19 skips.

## Future work

- [ ] Add dataset preflight reporting for templated prompt-length distributions.
