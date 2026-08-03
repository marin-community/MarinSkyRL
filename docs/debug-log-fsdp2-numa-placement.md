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

The pre-fix contract failed on Jupiter job 1216544, one four-GPU GH200 node, after initializing the production
FSDP2 policy-worker path. The worker still had the default memory policy after its pinned offload shards existed;
3,601 sampled pages were spread across LPDDR nodes 0, 1, and 2. This validates the test's red state without
relying on the separate production OOM.

The implementation installs an LPDDR-only hard memory policy in Ray's worker-process setup hook, before actor
threads are created, then repeats it alongside GPU-local CPU affinity in the actor constructor before CUDA or
FSDP2 initialization. The late lifecycle calls were removed because they cannot affect storage FSDP2 already
created.

## Future work

- [ ] Determine whether correcting host-memory placement changes wedge incidence; do not infer causality from
  placement alone.
