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

## Design notes

The completed buffer remains a bounded `asyncio.Queue`; producers use its standard capacity API while a shared
condition makes each all-buffer sweep atomic. Epoch queues are passed explicitly to tasks, restore, checkpoint binding,
and batch assembly, preserving epoch scope without ambient trainer state. Shared checkpoint records and the queue
provider protocol live in `async_rollout_state.py`, keeping callbacks independent from the concrete trainer module.

Workers select a prompt before acquiring capacity, so a worker waiting for a retry owns no slot. Accepted, stale,
cancelled, and failed attempts use distinct accounting transitions. Epoch validation requires the retry queue to be
empty before retiring it. Checkpoint persistence fails closed when trainer context, bound queues, or storage are
unavailable.

The group timestamp is `earliest_model_step`; scheduling time is its fallback when sampled-token capture is unavailable.
`minimum_captured_global_step` selects the oldest model version represented by any sample in the group.
Group routing is named `_classify_and_route_group`, making its freshness result explicit. Discard metrics assert the
invariant that returning a fresh batch must have inspected at least one completed group before calculating the rate.

## Future work

- None.
