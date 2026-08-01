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

CPU hosts collect and skip the suite because four CUDA devices are required. On a CUDA host, setup completes
only after every rank finishes three value-checked warmup rounds. The controller then starts a separate
45-second fault deadline and kills and reaps only its disposable torchrun process group if that deadline
expires. This keeps a reproduced process-group wedge from wedging the test runner.

## Future work

- [ ] Record dedicated-node GPU results in the pull request or linked execution artifact.
- [ ] If production settings exceed the harness deadline, add an out-of-process phase-progress watchdog and use
      this case as its red-to-green contract.
