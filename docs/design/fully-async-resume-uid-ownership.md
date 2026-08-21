# Fully asynchronous resume UID ownership

Fully asynchronous checkpoints distinguish three states for each dataset UID:

- **consumed**: a completed optimizer step used the group;
- **pending**: a completed group or retry request owns the UID but has not trained;
- **unscheduled**: no restored work owns the UID.

Consumed and pending UIDs have different accounting. Only consumed UIDs advance dataset-consumption metrics. Both sets
must be excluded when dataset iteration restarts from the beginning after a checkpoint resume.

## Decision

The generation-buffer artifact remains the source of truth for pending work. Resume derives the pending UID set from
the restored completed-group and retry queues, reserves those UIDs in the asynchronous dataloader, and then starts
generation workers. Partially generated groups are not checkpointed. Their UIDs remain unreserved and are generated
again after resume.

Reservations last for the current epoch. A retry keeps the same reservation. Successful training moves the UID into
the consumed set. The dataloader clears reservations when it resets for the next epoch.

Barrier assembly enforces unique UIDs independently of resume reservations. If completed work contains more than one
group for a UID, the barrier admits at most one eligible copy. It discards the other copies, releases their staleness
capacity, and does not retry them because the admitted copy already represents the dataset row. If every copy is
ineligible, the barrier schedules one retry and discards the rest.

This second check protects training from checkpoints written before pending UID reservations existed and from future
producer defects. The pre-training group assertion remains the final check on row alignment and algorithm-specific
group size.

## Rejected alternatives

Marking buffered groups as consumed would make consumption metrics include groups that never trained. Restoring the
stateful dataloader cursor would skip in-flight work that is not present in the checkpoint. Discarding the complete
generation buffer avoids duplicate work but loses valid rollouts at every wall-time restart.

## Required behavior

- A restored completed or retry UID is not issued from the restarted dataset iterator.
- A UID that was only in flight at checkpoint time is issued again.
- A training mini-batch contains no duplicate UID.
- Duplicate completed copies consume one batch slot, release surplus capacity, and schedule at most one retry.
- GRPO and other fixed-cardinality algorithms still receive their configured physical group size.
