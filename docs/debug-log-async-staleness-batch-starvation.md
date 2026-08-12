# Debugging log for async staleness batch starvation

Prevent `max_staleness_steps` from reducing the number of groups in a fully asynchronous training batch.

## Initial status

`FullyAsyncRayPPOTrainer` dequeues `policy_mini_batch_size` completed groups, discards each group whose recorded
staleness exceeds `max_staleness_steps`, and trains on any nonempty remainder. A run configured for 64 groups
therefore trained on 3 groups after discarding 61 stale groups.

## Hypothesis 1

Commit `59986b6f` disabled `_AsyncStalenessManager` capacity blocking and moved staleness enforcement to destructive
filtering after dequeue. Restoring capacity blocking and retaining late completed groups will preserve the configured
batch size while bounding the number of groups scheduled ahead of training.

## Changes to make

Add CPU regressions for the scheduling-capacity contract and for a 61-stale/3-fresh conversion batch. Run them
against the current implementation before changing production code.

## Results

Both regressions failed against commit `803e082c`: the capacity manager admitted the second group with
`max_staleness_steps=0`, and conversion returned 6 samples after receiving 64 two-sample groups. The production
change restores capacity blocking and converts all dequeued groups while reporting violations as metrics and one
warning per affected batch.

The lint review found a stale buffer-rationale comment and constant effective-batch metrics left behind by the
refactor. The comment now describes the bounded completed-output backlog, the constant metrics were removed, and
the shared reward log no longer reads the deleted effective-sample metric.

The focused async test set passed after the fix: 31 tests across staleness, buffer checkpoints, checkpoint-resume
boundaries, and training-batch replay. `uv run infra/pre-commit.py --changed-files --fix` also passed.

## Future work

- None.
