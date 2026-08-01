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

- ranks enter incompatible EP and FSDP subgroups;
- ranks enter WORLD collectives in different orders;
- one rank arrives after the collective deadline.

The parent test requires the gang to fail inside a 45-second wall-clock bound. If it does not, the parent kills
only that subprocess group and fails with the captured torchrun output. These tests disrupt their worker
processes by design; do not run them inside another distributed job or on a node serving unrelated work.
