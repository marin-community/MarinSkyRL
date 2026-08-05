# NCCL fault injection

This opt-in suite validates healthy expert-parallel communication and the failure bound for real
ProcessGroupNCCL collectives. It is intentionally outside `tests/gpu/gpu_ci/`, and its Python filename does not
match pytest's default discovery pattern.

Run it on an otherwise idle node with at least four GPUs:

```bash
cd skyrl-train
uv run --isolated --group dev --extra vllm \
  pytest -s tests/gpu/fault_injection/nccl_collective_contract.py
```

Each case launches a disposable four-rank `torchrun` gang. One checks a healthy EP-subgroup `all_to_all_single`
using production communicator settings and validates the values exchanged by every rank. The other cases
deliberately violate a collective contract:

- all ranks warm their EP `all_to_all_single` and FSDP `all_gather_into_tensor` communicators before ranks 0 and
  3 enter EP dispatch while ranks 1 and 2 enter FSDP gather under production communicator settings;
- one member never enters an EP-subgroup collective built through the production device-mesh path;
- one rank never enters a WORLD collective;
- one rank exits while its peers are blocked in a WORLD collective.

The controller allows up to 180 seconds for imports, WORLD initialization, and device-mesh construction. It
starts the fault timer only after all four ranks report ready, then requires the gang to fail within 45 seconds.
On either deadline it kills and reaps only that subprocess group under a second bounded timeout, and reports
the captured torchrun output. These tests disrupt their worker processes by design; do not run them inside
another distributed job or on a node serving unrelated work.

Run only the warmed production divergence experiment with:

```bash
cd skyrl-train
uv run --isolated --group dev --extra vllm \
  pytest -s tests/gpu/fault_injection/nccl_collective_contract.py \
  -k warmed_production_phase_divergence
```

The test is red if the production watchdog stack leaves the gang alive beyond 45 seconds. The controller then
kills and reaps the disposable subprocess group and includes the complete torchrun log in the failure. A pass
requires three successful EP/FSDP warmup rounds on every rank, all four ranks entering the mismatched phase, no
collective completing, and torchrun exiting nonzero before the controller deadline.

The EP-subgroup non-arrival, WORLD non-arrival, and rank-exit cases explicitly enable NCCL's nonblocking
communicator mode. The warmed phase-divergence case uses blocking production settings. Production workers
deliberately do not enable communicator nonblocking: NCCL permits ordinary calls to return `ncclInProgress`,
and PyTorch's EP `all_to_all_single` path does not handle that result. Keeping the healthy all-to-all and fault
teardown checks together prevents a fault-only result from being mistaken for a training-safe runtime default.

## Model collective schedule matrix

Run the healthy model-level matrix separately:

```bash
cd skyrl-train
python -m pytest -s tests/gpu/fault_injection/ep_fsdp_collective_matrix.py
```

Run that command inside the policy image so it uses the image's installed Torch, TorchTitan, and CUDA stack.
An isolated `uv` environment both requires network access to direct-URL dependencies and can resolve a different
CUDA build, so it is not an image-validation command.

The six cases launch serially in fresh four-rank gangs. They compose the real TorchTitan EP hooks and FSDP2
wrapping while varying live versus replayed routing, spread versus concentrated expert selection, and reentrant
activation checkpointing. The final case adds a two-second delay on rank 0 before layer 1; it observes eventual
completion and matching schedules without asserting a narrow runtime.

Every case must emit one `MODEL_COLLECTIVE_SCHEDULE_OK` record and exit zero within four minutes. A timeout kills
and reaps only that case's subprocess group and reports its complete captured output. This matrix is opt-in and
its filename deliberately remains outside pytest's default `test_*.py` discovery.

## Four-node EP/FSDP traffic

`multi_node_ep_fsdp_worker.py` is a direct torchrun worker for a 16-rank mesh: four ranks per node, EP4 within
each node, and FSDP4 across nodes. It rejects an allocation whose physical placement does not match that
contract. It then verifies FSDP all-gather, reduce-scatter, and all-reduce payloads at several sizes; alternates
those operations with node-local EP all-to-all; and repeats the cross-node traffic with one late-arriving rank
per node. The fixed workload uses 1, 8, and 32 MiB payloads, 32 alternating rounds, and two seconds of arrival
skew. Timings are evidence, not a performance gate.

Launch one torchrun agent on each node. For a four-node Slurm allocation, set a rendezvous address reachable
from every node and run:

```bash
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=29571
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c '
  torchrun --nnodes=4 --nproc-per-node=4 \
    --node-rank="$SLURM_NODEID" \
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    --module tests.gpu.fault_injection.multi_node_ep_fsdp_worker
'
```

Run from `skyrl-train/` in the same image and environment used by policy workers. The run passes only when one
rank prints `MULTI_NODE_EP_FSDP_TRAFFIC_OK` and every torchrun agent exits zero. A hang is a failure; the
process-group timeout is three minutes, so the enclosing cluster job needs a longer independent deadline.

## Four-node permanent phase divergence

`multi_node_nccl_contract.py` starts `multi_node_worker_bootstrap.py`, which applies the production Ray worker
setup before loading `multi_node_phase_divergence_worker.py`. Together they are the destructive counterpart to
the healthy traffic test. They use the same 16-rank EP4/FSDP4 placement and asynchronous watchdog mode. After
every rank completes three EP all-to-all and
inter-node FSDP all-gather warmup rounds, rank 0 enters an FSDP all-gather while the other 15 ranks enter EP
all-to-all. Twelve ranks complete unaffected EP groups and remain alive; rank 0 and its three EP peers wait for
participants that never arrive. The test passes only if the configured ProcessGroupNCCL deadline converts that
permanent phase divergence into a nonzero gang exit.

Run the pytest controller on the Slurm host directly from the batch process of an otherwise idle allocation
containing exactly four four-GPU nodes. Do not put the controller inside Apptainer and do not wrap pytest in
`srun`: it needs the host's `scontrol` and `srun`, and it owns the Slurm step and its bounded cleanup:

```bash
PYTHONPATH=<checkout>/skyrl-train /path/to/host/python -m pytest -s \
  <checkout>/skyrl-train/tests/gpu/fault_injection/multi_node_nccl_contract.py \
  --confcutdir=<checkout>/skyrl-train/tests/gpu/fault_injection \
  --node-agent-command-prefix='apptainer exec --nv --pwd / <policy.sif>'
```

The controller enters the policy image only for its remote node agents, so the 16 ranks use the production
PyTorch, CUDA, NCCL, and TorchTitan stack while orchestration remains on the Slurm host. It allows five minutes
for rendezvous, mesh construction, and healthy warmup. It starts a separate
two-minute fault deadline only after all 16 ranks report ready. Either deadline kills and reaps the disposable
Slurm step under a separate bounded reap deadline and includes its captured output in the failure. A passing run emits
16 warmup records, 16 readiness records with effective timeout and group membership, 16 fault-entry records,
and 12 unaffected-EP completion records; no blocked collective may return normally.

The controller injects the legacy `NCCL_BLOCKING_WAIT=1` setting seen in the TaskTrove launch environment before
starting the node agents. The production worker bootstrap must remove it before importing torch; every readiness
record therefore reports `blocking_wait=None`. This keeps the regression at the actual worker boundary and proves
that inherited launcher settings cannot switch ProcessGroupNCCL away from MarinSkyRL's asynchronous watchdog.
The controller also resolves the Slurm batch hostname to IPv4 before torchrun because Jupiter compute nodes do not
support the IPv6 addresses returned first by the cluster aliases.

Pass the policy-image command through `--node-agent-command-prefix`. The controller prepends it to every remote
node-agent command and invokes the image's `python`, so all four nodes use the same explicit policy runtime.
Omitting the prefix fails before launch. The host Python needs this checkout and its test dependencies, but it
is not the runtime under test. See
`.agents/ops/jupiter/` for the current Jupiter policy-runtime command and GPFS-safe launch procedure.
The `--confcutdir` boundary also prevents unrelated GPU fixtures from becoming host-controller dependencies.
