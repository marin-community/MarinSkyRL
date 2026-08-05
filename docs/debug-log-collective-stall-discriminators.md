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

Jupiter job 1246400 reran all three cases with the corrected completion boundary. The enqueued CUDA stall and
model schedule divergence passed. The pre-enqueue case terminated with the intended 16 before-enqueue, 15
after-enqueue mechanism, but its controller saw 19 before-enqueue strings: the withheld rank and its three peers
all emitted an explicit precondition record, then the three peers emitted the shared enqueue helper's record as
well. Restricting the explicit record to the withheld rank removes those three duplicates without changing the
fault.

Jupiter job 1246556 reran only the corrected pre-enqueue case at commit `0d90dfee`. It passed in 205.62
seconds with 16 before-enqueue records, 15 after-enqueue records, 12 healthy FSDP-subgroup completions, no
unexpected completion, and a nonzero gang exit. The final evidence is split across two runs: job 1246400
validated the enqueued CUDA stall and model schedule divergence, and job 1246556 validated pre-enqueue
non-arrival. The last code change only removes the pre-enqueue case's three duplicate records.

These tests establish distinct controlled signatures; they do not assign one to the natural TaskTrove wedge.
The next natural timeout needs flight-recorder state or equivalent before/after-enqueue evidence from ranks 1,
5, 9, and 13 before any of the three mechanisms can be selected.

Jupiter job 1246598 validated the complete post-review branch at commit `4d828288` against the same production
runtime. All three cases passed in 603.42 seconds. Their compact records were:

- enqueued CUDA stall: 16 before, 16 after, 12 unaffected FSDP completions;
- pre-enqueue non-arrival: 16 before, 15 after, 12 unaffected FSDP completions;
- model schedule divergence: 16 before, 16 after across different EP/FSDP operations, 12 unaffected FSDP
  completions.

Every blocked gang exited nonzero inside the independent controller deadline. The Slurm batch completed and
released all four nodes.

## Future work

- [ ] Compare these controlled signatures with the next natural timeout's flight-recorder artifacts.
