# Debugging log for colocated cross-node TP

Prevent colocated multi-GPU inference engines from spanning nodes and bound engine startup.

## Initial status

The colocated placement group contains one `{GPU: 1}` bundle per rank under soft `PACK`. Inference engines take contiguous bundle indices without checking their assigned nodes, so one TP group can span nodes. Engine readiness waits have no timeout.

## Hypothesis 1

Sorting the existing one-GPU placement bundles by their resolved node and GPU assignments before slicing them into engines preserves per-GPU identity and makes every valid TP/PP slice node-local.

## Changes to make

Add behavioral layout tests before changing placement construction and engine indexing. Add a bounded startup-wait regression using Ray as the external boundary.

## Results

The original tests failed at collection because neither the node-atomic layout API nor bounded startup wait existed. An initial whole-node-bundle design was rejected during self-review because Ray packs fractional GPU actors within a bundle and would lose rank-to-GPU identity. The final design retains one-GPU bundles and uses their resolved node/GPU ordering, matching the existing policy-worker path. Focused tests pass. Geometry validation rejects TP×PP sizes that cannot tile one node, and startup waits propagate actor errors as well as timeouts.

## Future work

- [ ] Change the pinned vLLM FlashInfer fallback so a workspace failure is agreed across every collective rank.
