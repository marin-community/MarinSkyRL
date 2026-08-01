# Debugging log for warmed collective divergence

Reproduce the observed mid-training split between expert-parallel token dispatch and FSDP parameter gathering
without relying on a cold communicator or a different collective type.

## Initial status

The opt-in GPU suite separately checks a healthy EP `all_to_all_single` under production communicator settings
and cold non-arrival faults under nonblocking diagnostic settings. Its EP non-arrival fault uses `all_reduce`
before any successful collective has initialized that subgroup. Those cases do not establish whether the
production watchdog stack terminates a rank divergence after both implicated communicators have carried work.

## Hypothesis 1

A four-rank test can reproduce the production collective-order split after warming both communicators. Ranks 0
and 3 entering their EP groups while ranks 1 and 2 enter their FSDP groups leaves every collective missing a
participant under the repository's `(fsdp=2, ep=2)` device mesh.

## Changes to make

Warm each rank's EP group with value-checked `all_to_all_single` calls and its FSDP group with value-checked
`all_gather_into_tensor` calls. Report setup complete only after three rounds and a WORLD barrier. Then inject
the phase split with communicator nonblocking disabled and require torchrun to exit nonzero within the existing
45-second fault deadline.

## Results

The four pre-existing cases pass on a GH200 node with the production Titan overlay: three destructive cases
terminate under nonblocking diagnostic settings, and the blocking EP all-to-all completes with correct values.
This validates the harness baseline but does not exercise a warmed, blocking phase divergence.

The expanded module collects on a CPU host and skips all five cases because four CUDA devices are required. The
new contract has not run on an NVIDIA node. Its bounded controller will kill and reap the disposable torchrun
process group if production settings reproduce the wedge.

## Future work

- [ ] Run the warmed production divergence case on a dedicated four-GPU node.
- [ ] If it exceeds the harness deadline, add an out-of-process phase-progress watchdog and use this case as its
      red-to-green contract.
