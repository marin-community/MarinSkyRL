# Iris nodes with hardware faults

Preserve the failing job's node-level evidence and notify the cluster operators through their active incident
channel. A GitHub issue alone is not an incident-response path.

Use `kubectl cordon` only with authority to change the shared cluster. A cordon removes a node from scheduling
for every workload, so it is appropriate for a confirmed node-level InfiniBand or hardware fault. It is not a
per-job exclusion mechanism. Track durable node-health detection and scheduler support in
[marin#7871](https://github.com/marin-community/marin/issues/7871).
