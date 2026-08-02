# Warmed collective divergence contract

This contract reproduces a mid-training split between expert-parallel token dispatch and FSDP parameter
gathering without relying on a cold communicator or a different collective type.

## Coverage gap

The opt-in GPU suite separately checks a healthy EP `all_to_all_single` under production communicator settings
and cold non-arrival faults under nonblocking diagnostic settings. Its EP non-arrival fault uses `all_reduce`
before any successful collective has initialized that subgroup. Those cases do not establish whether the
production watchdog stack terminates a rank divergence after both implicated communicators have carried work.

## Experiment design

Each rank warms its EP group with value-checked `all_to_all_single` calls and its FSDP group with value-checked
`all_gather_into_tensor` calls. Report setup complete only after three rounds and a WORLD barrier. Then inject
the phase split with communicator nonblocking disabled and require torchrun to exit nonzero within the existing
45-second fault deadline. Ranks 0 and 3 enter their EP groups while ranks 1 and 2 enter their FSDP groups, leaving
every collective missing a participant under the repository's `(fsdp=2, ep=2)` device mesh.

The controller starts the fault deadline only after all ranks report warmup completion. On expiry it kills and
reaps only its disposable torchrun process group, so a reproduced process-group wedge cannot wedge the test
runner.

## Interpretation

A pass means the production ProcessGroupNCCL configuration turns a warmed collective-order divergence into a
nonzero torchrun exit within 45 seconds. A harness-deadline failure means that configuration does not bound the
observed failure class. In that case, this test becomes the red contract for an out-of-process phase-progress
watchdog that converts stalled rank progress into process death.
