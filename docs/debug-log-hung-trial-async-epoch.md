# Debugging log for hung-trial-async-epoch

Convert one hung agent trial from a permanent job stall into a retried group.

## Initial status

Five agent trials out of 4,864 never returned, permanently stalling an async epoch for 9.5h.
Three independent defects each convert one hung trial into a dead job.

## Hypothesis 1 — `run_shard` has no deadline

`run_shard` awaits `self._generator.generate(sub_batch)` with no timeout. Wrap it in
`asyncio.wait_for` derived from the Harbor trial budget.

## Changes to make

- `rollout_coordinator.py`: store shard timeout from `override_timeout_sec`, wrap `generate()`
- `fully_async_trainer.py`: handle `TimeoutError` gracefully in `_run_generate_for_a_group_loop`
  (retry the items instead of `sys.exit(1)`)
- `fully_async_trainer.py`: bound `_get_admitted_generation_group_mini_batch` with adaptive deadline
- `fully_async_trainer.py`: bound `_next_generation_prompts` retry-queue wait
- `fully_async_trainer.py`: add `GenerationStalledError`, track step-time history
- `vllm_engine.py`: log `resume_generation`

## Results

(pending implementation)

## Future work
- [ ] Surface trainer liveness to controller (Part 5 from escalation)
- [ ] Make `critical_phase` report open-phase age (Part 3b from escalation)
