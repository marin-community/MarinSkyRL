# Debugging log for collective timeout contract

Ensure a rank-divergent policy collective cannot leave a training job running past the configured timeout.

## Initial status

Jupiter job 1141984 remained in policy backward for more than four hours. Python stacks placed most sampled
ranks in torchtitan's EP `all_to_all_single` and one rank in an FSDP all-gather. The configured worker process-
group timeout was 1,800 seconds.

Code inspection shows that the WORLD group receives the configured timeout at construction and the FSDP/EP/CP
device-mesh groups receive it through `_set_pg_timeout`. The implicated torchtitan call uses the same EP group.
The missing piece on Jupiter is `TORCH_NCCL_ENABLE_MONITORING`: without the monitor, a stuck ProcessGroupNCCL
watchdog thread has no independent process-termination bound.

## Hypothesis 1

The subgroup timeout was never applied to the group used by torchtitan EP.

## Changes to make

Exercise two real CPU/Gloo device meshes with ranks intentionally entering collectives on different groups.
Require both collectives to raise within the configured subgroup timeout.

## Results

The four-rank Gloo test sends ranks 0 and 3 into their EP groups while ranks 1 and 2 enter their FSDP
groups. Every group is missing one participant. All four collectives raise after the configured one-second
device-mesh timeout. Hypothesis 1 is refuted: the existing post-construction timeout reaches the exact
subgroups used by torch EP and FSDP.

## Hypothesis 2

The runtime environment enables async NCCL error handling but does not enable the independent watchdog monitor,
so a watchdog thread blocked in a CUDA/NCCL call can leave the process alive indefinitely.

## Changes to make

Add a failing runtime-environment contract test requiring monitoring and a heartbeat deadline independent of,
and no larger than, the collective timeout. Configure those variables for every Ray worker.

## Results

Both runtime-environment tests failed before the fix because neither `TORCH_NCCL_ENABLE_MONITORING` nor
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` was present. `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` was present, which only
handles an error after the watchdog observes one; it does not terminate a watchdog thread that has stopped
heartbeating.

The runtime environment now enables the independent monitor for every Ray worker and defaults to a five-minute
heartbeat deadline, capped by the collective timeout. The standard `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` remains
the override so existing launch configurations retain their intended heartbeat budget.

## Future work

- [ ] Run the same intentional mismatch with ProcessGroupNCCL in GPU CI; Gloo validates group scoping and timeout
      propagation but cannot emulate a watchdog thread blocked inside a CUDA API.
