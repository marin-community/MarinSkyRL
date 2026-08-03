# Debugging log for FSDP2 CPU-offload NUMA placement

FSDP2 must allocate persistent pinned CPU-offload storage in LPDDR attached to the policy rank's Grace CPU. HBM
NUMA nodes must not be eligible fallback nodes for host storage.

## Initial status

Live Jupiter policy workers created their initialization-time FSDP2 offload arenas under the default NUMA
policy. MarinSkyRL applied its NUMA preference only in the later explicit `offload_to_cpu()` lifecycle method.
Healthy jobs also carried tens to hundreds of GiB of host pages in HBM, so HBM placement is a standing exposure
and not a wedge discriminator.

## Hypothesis 1

The production policy-worker path allocates FSDP2's pinned CPU shards before installing a NUMA policy.

## Changes to make

Add an opt-in GH200 test that initializes the production Ray policy worker, samples physical pages from its
pinned FSDP2 parameters, and requires an LPDDR-only binding with GPU-local placement.

## Results

Pending.

## Future work

- [ ] Determine whether correcting host-memory placement changes wedge incidence; do not infer causality from
  placement alone.
