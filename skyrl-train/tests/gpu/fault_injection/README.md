# NCCL fault injection

This opt-in suite validates the failure bound for real ProcessGroupNCCL collectives. It is intentionally
outside `tests/gpu/gpu_ci/`, and its Python filename does not match pytest's default discovery pattern.

Run it on an otherwise idle node with at least four GPUs:

```bash
cd skyrl-train
uv run --isolated --extra dev --extra vllm \
  pytest -s tests/gpu/fault_injection/nccl_failure_contract.py
```

Each case launches a disposable four-rank `torchrun` gang and deliberately violates a collective contract:

- one member never enters an EP-subgroup collective built through the production device-mesh path;
- one rank never enters a WORLD collective;
- one rank exits while its peers are blocked in a WORLD collective.

The controller allows up to 180 seconds for imports, WORLD initialization, and device-mesh construction. It
starts the fault timer only after all four ranks report ready, then requires the gang to fail within 45 seconds.
On either deadline it kills and reaps only that subprocess group under a second bounded timeout, and reports
the captured torchrun output. These tests disrupt their worker processes by design; do not run them inside
another distributed job or on a node serving unrelated work.

The test obtains its communicator-level timeout environment from the same helper used by production Ray
workers. This is intentional: ProcessGroupNCCL's work watchdog does not cover a rank blocked before a
`WorkNCCL` exists, so the contract also requires abortable NCCL communicator operations.
