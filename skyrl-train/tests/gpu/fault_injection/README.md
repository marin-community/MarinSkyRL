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

- one member never enters an EP-subgroup collective built through the production device-mesh path;
- one rank never enters a WORLD collective;
- one rank exits while its peers are blocked in a WORLD collective.

The controller allows up to 180 seconds for imports, WORLD initialization, and device-mesh construction. It
starts the fault timer only after all four ranks report ready, then requires the gang to fail within 45 seconds.
On either deadline it kills and reaps only that subprocess group under a second bounded timeout, and reports
the captured torchrun output. These tests disrupt their worker processes by design; do not run them inside
another distributed job or on a node serving unrelated work.

The failure cases explicitly enable NCCL's nonblocking communicator mode because it makes communicator-level
non-arrival abortable. Production workers deliberately do not enable that mode: NCCL permits ordinary calls to
return `ncclInProgress`, and PyTorch's EP `all_to_all_single` path does not handle that result. Keeping the
healthy all-to-all and fault teardown checks together prevents a fault-only result from being mistaken for a
training-safe runtime default.
