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

The focused async test set passed after the fix across staleness, buffer checkpoints, checkpoint-resume boundaries,
training-batch replay, and generator timestamp utilities. `uv run infra/pre-commit.py --all-files --fix` also passed.

## Hypothesis 2

Preserving every completed group keeps batch cardinality stable but trains on rollouts outside the configured
staleness cap. A stale attempt must be replaced by a new attempt for the same dataset row. Replacing it with the
next row would exhaust the epoch early and leave the stale row unconsumed.

## Changes to make

Retain the source prompt with each completed group. Sweep the entire completed-output queue at each batch boundary,
send every stale prompt to a retry queue, and wait until `policy_mini_batch_size` fresh groups are available. Keep
generation workers alive after the dataset iterator is exhausted so they can service retries.

## Results

The new regressions failed against `65f53029`: no full-buffer sweep or replacement wait existed. The final path
drains all completed groups under a shared buffer condition, retries every stale group with its original prompt,
and blocks the optimizer until a full fresh batch is available. Producers re-check staleness after waiting for
buffer space, so a completed group blocked outside the queue cannot bypass the sweep.

The checkpoint artifact now preserves pending retries. Native SkyRL Gym and step-wise generators use the minimum
captured sample step as the group's age; Terminal Bench already used the earliest trial start. The focused CPU set
passed 96 tests after the change. Buffer checkpoint persistence also fails closed: retry state is snapshotted on the
event loop, and a storage error now stops checkpoint handling instead of logging a warning and continuing.

## Lint-review disposition

The repeated queue-drain loops now use one helper, the constant submission-slot return value was removed, and the
step-wise generator comment now describes the earliest-sample timestamp. The review suggested moving the stale-batch
cluster out of the trainer, but those methods deliberately coordinate trainer-owned `global_step`, metrics, queue
telemetry, and `_AsyncStalenessManager` accounting; moving them would add a stateful facade without reducing coupling.
Submission-slot cleanup remains explicit because completion, stale rejection, cancellation, and worker failure have
different accounting transitions; an async context manager would still need the same outcome-specific branches.

## Future work

- None.
