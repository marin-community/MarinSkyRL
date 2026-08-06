# Distributed debug mode

Use distributed debug mode for a canary investigating multi-rank stalls:

```yaml
trainer:
  debug_mode: distributed
```

That config works through non-Iris launchers, including Jupiter `hpc.launch`. Iris also offers a convenience
flag that resolves to the same contract:

```bash
marinskyrl ... --debug-mode distributed
```

Normal mode keeps MarinSkyRL at `INFO`, NCCL at `WARN`, and expensive phase/timing/stack recording off. Debug
mode keeps MarinSkyRL at `INFO` and uses NCCL `INFO` only for communicator initialization, bootstrap,
environment, network, topology, and tuning. Per-collective text logging is excluded; the bounded PyTorch flight
recorder captures that history instead.

The preset enables per-rank phase records, NCCL desynchronization diagnostics, collective timing, C++ stacks,
PyTorch C++ informational logs, fast symbolization, dump-on-timeout, process manifests, Python fault handling,
and the policy host-memory monitor. It intentionally does not enable `CUDA_LAUNCH_BLOCKING` or
`TORCH_DISTRIBUTED_DEBUG=DETAIL`, because those settings change synchronization and can hide or create timing
failures. See the [PyTorch flight-recorder guide](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html)
and [NCCL logging reference](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#environment-variables)
for the underlying controls.

For a local checkpoint path, artifacts land in a sibling `debug/` directory next to `checkpoints/`. This is the
durable GPFS path on Jupiter. An object-store checkpoint path uses a job-scoped node-local staging directory;
the Iris task runtime periodically and finally uploads it under:

```text
<rendezvous>/debug_artifacts/<node-id>/
```

Each Iris upload writes `sync-manifest.json` with every copied or budget-rejected file. A single file is capped
at 512 MiB and one node sync at 2 GiB.

## Jupiter acceptance test

The opt-in contract runs two sequential two-node, one-GPU-per-node gangs. The healthy gang must complete. The
second first warms its NCCL communicator on both ranks, then withholds rank 1 from the next collective and must
fail within the independent controller deadline. Warming is load-bearing: withholding a rank from the first
collective can block inside lazy communicator creation before a `WorkNCCL` exists for the process-group timeout
to inspect. The test is green only when both gangs terminate and every process manifest, NCCL setup log, healthy
rank outcome, withheld-rank receipt, and failed-run flight-recorder dump exists beneath the explicit GPFS
destination. The timed-out rank exits by `SIGABRT` after dumping, so the controller manifest records its nonzero
return instead of expecting unreachable post-abort Python code to write a receipt.

Run the host controller from a two-node allocation. Use the current production runtime selected through
`.agents/ops/jupiter/production-runtime.env`; do not resolve a new Python/CUDA environment on compute nodes:

```bash
CHECKOUT=/e/scratch/<project>/<user>/worktrees/<branch>
ARTIFACT_ROOT=/e/scratch/<project>/<user>/debug-contracts
. "$CHECKOUT/.agents/ops/jupiter/production-runtime.env"
POLICY_PYTHONPATH="$CHECKOUT:$CHECKOUT/skyrl-train:$POLICY_PYDEPS"

cd "$CHECKOUT/skyrl-train"
PYTHONPATH="$CHECKOUT:$CHECKOUT/skyrl-train" "$HOST_PYTHON" -m pytest -s \
  tests/gpu/fault_injection/distributed_debug_artifact_contract.py \
  --confcutdir=tests/gpu/fault_injection \
  --debug-artifact-root="$ARTIFACT_ROOT" \
  --node-agent-command-prefix="apptainer exec --nv --pwd / --overlay $POLICY_OVERLAY:ro \
    --env PYTHONPATH=$POLICY_PYTHONPATH $POLICY_SIF"
```

The controller must be the batch process, not an `srun` child: it owns and reaps the two sequential Slurm
steps. Preserve the artifact root when reporting the result.
