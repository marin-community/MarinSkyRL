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
The follow-up review led to directly binding each epoch's queues to the checkpoint callback, a named checkpoint-state
return type, typed queue draining, and outcome-specific routing and inspection names.
The shared generated-group and checkpoint-state records live in `async_rollout_state.py`, which keeps callback typing
independent of the trainer module. The obsolete trainer queue attribute and redundant queue-count test were removed.
Queue snapshots now return that named checkpoint-state record directly. The generation worker retains its explicit
outcome-specific slot transitions because accepted, stale, cancelled, and failed attempts require different accounting.
`HasCapturedGlobalStep` remains structural to keep the generator utility module from importing `AgentLoopOutput` back
from `skyrl_gym_generator`, which already depends on the utility module.
An unbound checkpoint callback now raises instead of silently omitting the buffer artifact. The staleness counters are
initialized with the trainer, stale-acceptance documentation was corrected, and the obsolete rejection method was
removed. Batch assembly stays together because the condition lock must cover the all-buffer sweep, retry enqueue, and
fresh-overflow restoration as one atomic queue transition; capacity reconciliation then occurs immediately afterward.
The fully async tutorial now documents replacement-and-wait rather than the obsolete accept-and-train policy. Missing
trainer context also fails checkpoint persistence, and worker cancellation uses the manager's locked slot release.
The tutorial's checkpoint section now covers completed groups and pending retries. The stale partition helper was renamed
to expose its retry side effect, and post-drain buffer telemetry that could only report zero was removed. The completed
buffer remains a bounded `asyncio.Queue`: producers use its capacity API and condition notifications together, and a
deque conversion would replace a standard bounded primitive with hand-maintained capacity state without changing the
atomic sweep requirement.
Workers now select or wait for a prompt before acquiring a submission slot. A worker blocked on the retry queue therefore
owns no slot, eliminating the cancellation double-decrement race. The shared queue-provider protocol moved beside the
shared async state records, and tutorial counter definitions now account for discarded attempts.
All older tutorial descriptions now match the sweep, retry, and wait policy. Producer and consumer paths share one
stale-routing helper with an explicit freshness outcome, preventing their retry behavior from drifting.
The worker tutorial now states that completion is rechecked and routed, and a comment that only restated the
earliest-step helper call was removed.
The buffer tutorial now describes both queues, the shared condition, the atomic sweep, producer re-check, and checkpoint
snapshot. Regression-test setup is shared through one factory so the two batch-policy tests cannot drift.
The stale/fresh split now returns a named partition, and its one-use staleness comparison is inlined. Epoch queues stay
explicitly threaded through the trainer methods: this preserves epoch scope and avoids ambient mutable trainer state,
while ensuring tasks, restore, checkpoint binding, and batch assembly share the same object.
The callback's queue provider is now explicitly typed and discard-only metrics are named accordingly. The provider
protocol remains necessary to keep callbacks independent from the concrete trainer module. The named freshness
partition remains because positional same-type lists were a prior review finding. Batch assembly keeps queue mutation
under the condition and reconciles the exact discarded count immediately outside it; separating those phases would
require another state carrier without reducing the synchronization surface.
The bounded-buffer comment now names the condition wait. Epoch validation asserts that no stale-group retries remain,
so an unconsumed source row cannot be silently dropped when queues are retired.
The group timestamp is now named `earliest_model_step`, matching its primary inference-derived meaning; the schedule
step remains only its fallback when a generator cannot report sampled-token steps.
The submission counter documentation now describes capacity accounting rather than the obsolete failure-ratio metric.
All rollout counters now consistently use groups, and the regression helper names its timestamp `earliest_model_step`.
The queue-provider protocol remains a deliberate dependency boundary: moving the concrete queue into the state module
would make that leaf own `asyncio.Condition` synchronization behavior rather than just shared state records.
Discard/inspection accumulation is named `_record_discard_scan`. The bounded `asyncio.Queue` remains intentional:
producer capacity uses its standard `full`, `maxsize`, `qsize`, and nonblocking mutation APIs, while the shared condition
coordinates the atomic all-buffer sweep; a deque would require duplicating that bounded-queue behavior manually.
The tutorial's cap definition now uses the earliest captured model step. The named freshness partition remains an
intentional response to the review's earlier finding that a same-typed two-list tuple was positionally swappable.
The timestamp helper is named `minimum_captured_global_step`, distinguishing minimum model-version value from temporal
capture order while preserving the oldest-model sample policy.

## Future work

- None.
