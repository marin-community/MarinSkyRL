# Debugging log for RL watcher TIS metrics

Determine why active RL rows show unavailable TIS diagnostics and make the status table distinguish disabled
TIS from an enabled run that failed to emit metrics.

## Initial status

Four active Grug runs showed `TIS exact=—; TIS r=—` while ordinary policy log-ratio diagnostics were present.

## Hypothesis 1

The watcher does not recognize the namespace emitted by current policy workers.

## Results

Refuted. The watcher recognizes `policy/tis/log_ratio_abs_mean` and `policy/tis/imp_ratio_mean`, matching the
trainer's `policy/` prefix over the worker's `tis/*` status keys. Existing tests cover that namespace.

The current finelogs contain no `policy/tis/*` keys. Their resolved Hydra configurations consistently report
`use_tis: false` and `tis_imp_ratio_cap: -1.0`, so the policy workers correctly skip TIS diagnostics. The
available `policy/log_ratio_abs_*` fields measure policy movement and are not TIS rollout-to-training ratios.

## Hypothesis 2

The unconditional TIS cells make disabled TIS look like a parser failure.

## Changes to make

Parse the resolved `use_tis` value from the finelog. Render `TIS disabled` when it is false and no TIS metrics
exist. Render `TIS enabled; metrics missing` when it is true and no metrics exist. Continue displaying true TIS
metrics when they are present, including for older logs without a captured feature flag.

## Results

The two regression tests failed before the change because the table rendered `TIS exact=—; TIS r=—` for
both disabled TIS and an enabled run with missing diagnostics. They pass after the watcher carries the resolved
feature state into trend rendering.

Applying the patched parser to the captured `ctrl-grug` finelog returned `step=2`, `tis_enabled=False`, and
`TIS disabled`. The complete watcher and shared-metrics test set passed.

## Future work

- [ ] Enable TIS explicitly in a campaign configuration when truncated importance sampling is intended to
  affect training; changing that algorithm setting is outside the watcher.
