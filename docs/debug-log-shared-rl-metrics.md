# Debugging log for shared RL metrics

Unify training-metric extraction so the Iris watcher and offline RL analysis report the same
step, reward, policy loss, and gradient norm for standard and agentic jobs.

## Initial status

The watcher owns a one-record `WANDB_MIRROR` parser and metric aliases, while
`infra.rl_cleanup.parse_skyrl_metrics` owns a separate standard-run parser. The offline
`infra.rl_analysis` pipeline shells out to the latter. Before a job emits its first completed
training event, the watcher can report progress only from the progress-bar fallback; a
complete event carries the status fields.

## Hypothesis 1

A shared dependency-neutral parser in `infra.rl_metrics` can make live status and offline
analysis consume one behavior contract without coupling either consumer to the other's UI.

## Changes to make

Add a behavior-level regression test using a real standard-run `WANDB_MIRROR` shape. Then
move ANSI stripping, JSON extraction, step fallback, and canonical metric aliases into
`infra.rl_metrics`, and adapt both consumers.

## Results

The regression test failed at collection before the shared module existed. After the
change, the shared behavior tests pass 2/2 and the existing offline metrics and behavioral
analysis tests pass 7/7. The live watcher keeps its progress-bar fallback, but uses the
latest shared structured record as soon as a completed step is emitted.

## Lint review

The review's unused `infra.rl_analysis` re-export and redundant `__all__` findings were
fixed by keeping the package initializer empty; the analysis implementation imports the
shared parser directly. The watcher now returns a named result, calls `metric_value`
directly, and shares malformed-line message construction with offline analysis.

The ANSI helper remains in `infra.rl_metrics`: the other apparent copies serve different
deployment boundaries. `iris_ops.py` is an independently executable operations script
with a broader terminal-control regex, while the nightly gate is installed and run from
the independent `skyrl-train` package. Importing the repository-level analysis module
from either would break that isolation.

## Future work

- [ ] Consider streaming parsing if multi-million-line complete logs become a local memory bottleneck.
