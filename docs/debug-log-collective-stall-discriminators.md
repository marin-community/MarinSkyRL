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

Pending the on-demand four-node Jupiter run.

## Future work

- [ ] Compare these controlled signatures with the next natural timeout's flight-recorder artifacts.
