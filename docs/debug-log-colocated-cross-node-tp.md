# Colocated inference-engine placement

Colocated multi-GPU inference engines use the policy placement group's one-GPU bundles. Each engine's tensor-
and pipeline-parallel ranks must remain on one node because vLLM's cross-node FlashInfer symmetric-memory
rendezvous cannot pass file descriptors between hosts.

## Layout

Ray's `PACK` strategy does not guarantee that adjacent bundle indices are on the same node. After Ray assigns the
placement group, MarinSkyRL resolves every bundle's node and GPU, sorts the indices by `(node_id, gpu_id)`, and
slices that ordering into TP×PP groups. The TP×PP size must divide the number of GPUs assigned per node.

The one-GPU bundle shape is retained because colocated inference actors request fractional GPUs. Replacing it
with whole-node bundles would allow several actors to share one physical GPU and lose the one-rank-per-GPU
mapping.

## Startup failure

Engine readiness and the initial sleep transition share `generator.engine_init_timeout_seconds`. A timeout kills
the full engine actor gang and reports the pending actor indices. This prevents a live-but-hung vLLM rank from
leaving the training job in `running` indefinitely.

The vLLM FlashInfer fallback still handles workspace failures per rank. Collective agreement on disabling the
fusion remains dependency work; node-atomic placement prevents MarinSkyRL's supported colocated geometry from
entering that cross-node failure path.
