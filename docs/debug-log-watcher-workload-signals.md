# Debugging log for watcher workload signals

Make CoreWeave RL workload signals describe current-attempt failures without flagging benign warning tracebacks.

## Initial status

The watcher scans the last 2 MB of a cumulative Iris finelog for raw error tokens. It reports vLLM warning-level import probes and errors from earlier launches under the same job name, and returns only the matched token.

## Hypothesis 1

Line-aware severity filtering plus the rank-zero Iris setup marker can isolate actionable failures without suppressing unlevelled driver exceptions.

## Changes to make

Add regressions for warning-level tracebacks, a real driver exception, and a prior-attempt exception followed by a new setup boundary. Change `terminal_signal` to scan current-attempt lines in log order and return the complete matching line.

## Results

Confirmed. All three regressions failed against the token-only tail scanner. The line-aware scanner ignores explicit non-error levels, resets at the latest rank-zero Iris setup boundary when it is present in the tail, and returns the complete first matching line from the current attempt.

## Future work

- [ ] Confirm the next status sweep clears healthy vLLM jobs while retaining real current-attempt failures.
