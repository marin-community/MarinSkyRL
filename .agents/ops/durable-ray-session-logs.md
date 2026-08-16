# Debugging log for durable Ray session logs

Keep per-actor Ray session logs available for postmortems while retaining lifecycle-managed storage for transient RL
artifacts.

## Initial status

The launcher derives its rendezvous root under `tmp/ttl=Nd`. Task runtime stores Ray session logs below that root, so the
only copy of worker and actor tracebacks expires with the rendezvous data. The log-sync utility also rewrites any
rendezvous path without an `iris/` segment as an `iris/` object key, making the new lifecycle path undiscoverable.

## Hypothesis 1

Ray logs need an explicit durable storage root independent of rendezvous and spill data. Passing that root through the
existing task command keeps lifecycle policy centralized in the launcher and avoids deriving storage inside each node.

## Changes to make

Add a durable Ray-log root to resolved storage, pass it to task runtime, and use it for periodic and final Ray-log
uploads. Keep termination and distributed-debug artifacts under rendezvous storage because this escalation concerns the
otherwise-unavailable Ray actor logs.

## Results

The launcher derives a user-owned root and passes it to task runtime. Periodic, signal, and final uploads use that root
while rendezvous, termination, and distributed-debug artifacts retain their existing lifecycle policy.

## Hypothesis 2

Object-key normalization should remove a URI scheme and bucket without inventing an `iris/` prefix for explicit object
keys. Historical shorthand paths can retain the old `iris/` default.

## Changes to make

Parse full object-store URIs structurally and preserve their path. Teach the sync tool to accept the new Ray-log root
directly and derive it from the launcher banner, while retaining legacy rendezvous and agentic-run resolution.

## Results

URI parsing produces the actual `tmp/ttl=...` object key for existing runs. Current durable paths can be supplied directly
or read from the launcher banner, and the existing agentic and rendezvous layouts remain discoverable.

## Future work

- [ ] Decide whether termination and distributed-debug artifacts also need durable retention.
