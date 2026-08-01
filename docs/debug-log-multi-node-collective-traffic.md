# Multi-node EP/FSDP collective traffic contract

The opt-in GPU worker exercises a four-node policy mesh with 16 ranks. It verifies
the physical topology before interpreting collective results: every EP4 group must
occupy one host, and every FSDP4 group must span four hosts.

## Failure contract

The worker verifies the contents of FSDP all-gather, reduce-scatter, and all-reduce
payloads at 1, 8, and 32 MiB. It then runs 32 rounds that alternate those cross-node
operations with node-local EP all-to-all. A final phase delays one member of every
FSDP group by two seconds before another gather and reduce-scatter.

Any placement mismatch, corrupted payload, collective timeout, or nonzero torchrun
agent exit fails the run. The worker applies a three-minute timeout to WORLD and
device-mesh process groups. The enclosing cluster job must provide its own longer
deadline because no controller process can reap torchrun agents on other nodes.

## Scope

This contract isolates healthy NCCL traffic and bounded arrival skew. It does not
inject permanent rank divergence, run the full policy model, set a performance
threshold, or identify the cause of a production straggler. A pass establishes
that the selected image and allocation can execute the measured topology and
traffic patterns without corruption or a hang.

Run the worker only through torchrun's module mode, as documented in
`skyrl-train/tests/gpu/fault_injection/README.md`. Module mode keeps shared test
utilities importable without depending on an ambient `PYTHONPATH`.
