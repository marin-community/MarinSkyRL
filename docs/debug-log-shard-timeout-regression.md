# Debugging log for shard timeout regression

Prevent the rollout fan-out backstop from terminating healthy fully asynchronous agentic runs.

## Initial status

PR #375 set the group-level deadline to twice one Harbor agent deadline and let the resulting
`asyncio.TimeoutError` reach a trainer handler that exits the process. Jupiter jobs consistently terminated at
about 3,600 seconds while other rollout groups were completing normally.

## Hypothesis 1

The deadline omits Harbor's configured retry attempts and backoff, so a valid long-tail group can exceed it.

## Changes to make

Add a behavior test for the deadline derived from the agent deadline, attempt count, and capped exponential
backoff. Add an independent `rollout.fanout.shard_timeout_seconds` override.

## Results

The old deadline was 3,600 seconds for the campaign. The corrected derived deadline is 7,620 seconds:
four 1,800-second attempts plus 60, 120, and 240 seconds of backoff. The independent override leaves the
1,800-second agent deadline unchanged.

## Hypothesis 2

An outer timeout bypasses Harbor's `AgentTimeoutError` retry and error-treatment policy, then the fully
asynchronous trainer converts that group-local event into a process exit.

## Changes to make

Exercise a long-tail generator through the coordinator timeout boundary. Reclassify the outer timeout as
`AgentTimeoutError`, honor the YAML retry filter and retry count, return the configured terminal failure output
when retries are disabled or exhausted, and keep the generation worker alive.

## Results

The regression test failed before the fix because no shard-timeout policy existed. After the fix, a retryable
outer timeout completes on the next attempt and reports `generate/outer_agent_timeouts=1`; an excluded timeout
returns a terminal `AgentTimeoutError` group with the configured baseline exclusion instead of raising to the
trainer. The focused asynchronous generation, staleness, retention, and configuration tests pass (57 tests).

## Future work

- [ ] None identified.
