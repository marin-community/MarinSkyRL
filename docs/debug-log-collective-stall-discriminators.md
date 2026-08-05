# Debugging log for collective-stall discriminators

Determine which of three mechanisms can produce the one-subgroup loss of progress observed in TaskTrove policy
training, and establish a repeatable signature for each mechanism under the production EP4/FSDP4 geometry.

## Initial status

The natural incident records matching EP, FSDP, and WORLD sequence numbers when every rank entered backward.
Only ranks 1, 5, 9, and 13 then failed to exit. Existing fault tests prove that permanent rank non-arrival and
forced EP/FSDP phase divergence terminate under the corrected watchdog environment, but they do not prove
whether every implicated rank enqueued the same next collective.

## Hypothesis 1

Before-enqueue and after-enqueue evidence around an asynchronous collective can distinguish a collective that
was scheduled on every subgroup rank from a rank that stopped before scheduling it. An autograd hook that
issues a different subgroup collective can isolate model-caused schedule divergence without involving rollout,
checkpoint, or optimizer state.

## Changes to make

- Warm EP all-to-all and FSDP all-gather communicators on 16 ranks in the production four-node mesh.
- Run each fault in a fresh gang: the same FSDP work enqueued behind a CUDA stall, one missing FSDP enqueue, and
  an extra EP collective issued by one rank from a model backward hook.
- Record machine-readable markers on both sides of every injected enqueue.
- Require the production ProcessGroupNCCL timeout to terminate each gang under an independent controller
  deadline and bounded reap.

## Results

Jupiter job 1246330 ran all three cases on four GH200 nodes against the production Torch 2.11/CUDA 13 SIF and
Titan overlay. Every case warmed its communicators, emitted the expected before-enqueue and after-enqueue
counts, and terminated nonzero. The controller still marked every case failed because the worker emitted an
unexpected-completion record immediately after `WorkNCCL.wait()`.

This refuted the harness's first completion oracle. With asynchronous error handling, `wait()` establishes a
CUDA stream dependency but does not by itself prove CPU-visible GPU completion. The worker must synchronize the
device before it can report that a collective completed. The next run adds that synchronization without
changing the fault inputs or deadlines.

## Future work

- [ ] Compare these controlled signatures with the next natural timeout's flight-recorder artifacts.
