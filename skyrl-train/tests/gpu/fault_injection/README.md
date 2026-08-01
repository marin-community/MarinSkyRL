# NCCL fault injection

This opt-in suite validates healthy expert-parallel communication and the failure bound for real
ProcessGroupNCCL collectives. It is intentionally outside `tests/gpu/gpu_ci/`, and its Python filename does not
match pytest's default discovery pattern.

Run it on an otherwise idle node with at least four GPUs:

```bash
cd skyrl-train
uv run --isolated --extra dev --extra vllm \
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
uv run --isolated --extra dev --extra vllm \
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
uv run --isolated --extra dev --extra ep \
  pytest -s tests/gpu/fault_injection/ep_fsdp_collective_matrix.py
```

The six cases launch serially in fresh four-rank gangs. They compose the real TorchTitan EP hooks and FSDP2
wrapping while varying live versus replayed routing, spread versus concentrated expert selection, reentrant
activation checkpointing, and a two-second delay on rank 0 before layer 1. The delay case observes eventual
completion and matching schedules; it does not assert a narrow runtime.

Every case must emit one `MODEL_COLLECTIVE_SCHEDULE_OK` record and exit zero within four minutes. A timeout kills
and reaps only that case's subprocess group and reports its complete captured output. This matrix is opt-in and
its filename deliberately remains outside pytest's default `test_*.py` discovery.
