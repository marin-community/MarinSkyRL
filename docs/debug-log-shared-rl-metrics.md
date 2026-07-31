# Debugging log for shared RL metrics

Unify training-metric extraction so the Iris watcher and offline RL analysis report the same
step, reward, policy loss, and gradient norm for standard and agentic jobs.

## Initial status

The watcher owns a one-record `WANDB_MIRROR` parser and metric aliases, while
`infra.rl_cleanup.parse_skyrl_metrics` owns a separate standard-run parser. The offline
`infra.rl_analysis` pipeline shells out to the latter. The live job cited in the report had
not emitted a completed step when the status table was captured; its first complete
`WANDB_MIRROR` record appeared later and contains all four requested fields.

## Hypothesis 1

A shared parser and typed snapshot in `infra.rl_analysis` can make live status and offline
analysis consume one behavior contract without coupling either consumer to the other's UI.

## Changes to make

Add a behavior-level regression test using a real standard-run `WANDB_MIRROR` shape. Then
move ANSI stripping, JSON extraction, step fallback, and canonical metric aliases into
`infra.rl_analysis`, and adapt both consumers.

## Results

The regression test failed at collection before the shared module existed. After the
change, the shared behavior tests pass 2/2 and the existing offline metrics and behavioral
analysis tests pass 7/7. The live watcher keeps its progress-bar fallback, but uses the
latest shared structured record as soon as a completed step is emitted.

## Future work

- [ ] Consider streaming parsing if multi-million-line complete logs become a local memory bottleneck.
