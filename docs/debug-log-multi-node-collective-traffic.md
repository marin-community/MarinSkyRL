# Debugging log for multi-node EP/FSDP traffic

Build an opt-in GPU contract for the production-shaped four-node policy mesh. The
test must prove that EP traffic is node-local and FSDP traffic crosses all four
nodes before interpreting any collective result.

## Initial status

The existing schedule tests use one four-GPU node. They validate collective
ordering and bounded teardown, but their EP2/FSDP2 process groups cannot exercise
the four-node path observed in the Jupiter incident. The incident directly sampled
11 ranks in EP all-to-all and one rank in FSDP all-gather; the unsampled ranks do
not justify a 15-to-1 claim.

## Hypothesis 1

A four-node worker can isolate the physical path without a full trainer by using
the same `create_device_mesh` topology and real NCCL collectives. Hostname
inventory makes the path an asserted precondition: EP4 must remain on one host,
while FSDP4 must include four hosts.

## Changes to make

Add a direct torchrun worker outside default pytest discovery. Exercise FSDP
all-gather, reduce-scatter, and all-reduce at multiple payload sizes, interleave
them with EP all-to-all, and add bounded arrival skew. Verify every payload and
print one structured success marker. Document a generic four-node Slurm launch.

## Results

The first direct CLI check found that importing a helper from the repository's
`tests` package depended on `PYTHONPATH`; installing `skyrl_train` alone does not
make that package importable. The documented launcher now uses torchrun's
`--module` mode from `skyrl-train/`, which makes the test utilities importable
without duplicating their environment policy. The module's `--help` path succeeds
from a clean environment.

The collective schedule and device-mesh timeout CPU tests pass (4 tests). Ruff
check and format pass. The behavioral result still requires a four-node,
four-GPU-per-node allocation.

## Future work

- [ ] Run the exact branch revision on four GH200 nodes in the policy image.
- [ ] Add an exact-topology divergence case only after the healthy traffic
      contract distinguishes code failure from fabric or placement failure.
