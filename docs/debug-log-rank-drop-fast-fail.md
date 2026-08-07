# Debugging log for distributed worker fast-fail

Determine whether a dead Ray policy actor can leave peer actors blocked until the NCCL timeout, then add the smallest safe failure contract that prevents a known rank error from leaving its worker group alive.

## Initial status

Four 67B/32k runs timed out in NCCL collectives after 1,800 seconds. The preserved logs show the timeout and later `ActorDiedError` or `ActorUnavailableError`, but do not show that an actor died before the collective stalled. Cgroup pressure warnings preceded some failures.

## Hypothesis 1

`ray.get()` over a list waits for every reference even when one actor has already died, so a blocked peer delays the actor error.

## Changes to make

Run two local Ray actors. Terminate one process while the other method remains blocked, then call `ray.get()` on both references.

## Results

Refuted. Ray raised `ActorDiedError` immediately while the peer reference remained unresolved. An actual actor death already reaches the trainer without waiting for the peer task or the NCCL timeout. The incidents are consistent with an actor process remaining alive while its CUDA/NCCL task stops progressing.

## Hypothesis 2

Once one distributed actor task returns an exception, collecting references individually can identify its rank and terminate the full actor group before propagating a labelled error. This gives whole-job checkpoint retry a clean failure boundary without attempting to admit one replacement rank to an existing NCCL communicator.

## Changes to make

Add a rank-aware worker-group collector, use it for policy and critic `ppo_train` calls, and test it with one actor raising an OOM exception while a peer remains blocked.

## Results

The rank-aware collector surfaced both an actor task `OutOfMemoryError` and an actor process exit while a peer task remained unresolved. It terminated the peer actor, preserved the failed actor's mesh position in a picklable `WorkerGroupTaskError`, and left successful dispatch collection unchanged. The focused dispatch and trainer suites passed with 22 tests. The full launcher and trainer CPU gate passed with 1,236 tests and 21 skips.

This does not shorten a stall where every actor process and task remains alive. Those failures still rely on the configured ProcessGroupNCCL timeout until a device-progress signal can distinguish a hang from a long valid model operation.

## Future work

- [ ] Distinguish live CUDA-stream stalls from long valid model operations. Actor liveness cannot make this distinction.
- [ ] Evaluate CUDA progress sentinels and NCCL RAS under issue #327.
