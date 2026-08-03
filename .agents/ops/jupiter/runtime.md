# Jupiter RL runtime

Select a runtime from the launched configuration, then verify it in the allocation before interpreting a test
result. Do not choose a SIF from its filename alone.

## Production policy runtime

Versioned Jupiter artifact references live in `production-runtime.env`, separate from this procedure so an
ops-only change can advance them without rewriting the runbook. Source the manifest, verify every referenced
path, and compare it with the launch configuration before use:

```bash
set -a
. .agents/ops/jupiter/production-runtime.env
set +a
stat "$POLICY_SIF" "$POLICY_OVERLAY" ${POLICY_PYDEPS//:/ }
```

The policy SIF carries the modern Torch/CUDA/vLLM stack and GH200 extensions. The Titan overlay shadows packages
in the SIF, so the manifest's pydeps order is part of the runtime contract. Validate the imported symbol under
the complete overlay and `PYTHONPATH`, not by comparing package metadata. Use `CONTEXT_PARALLEL_POLICY_SIF`
only when the configuration requires context-parallel routed-expert capture, and verify that capability before
launch.

Older torch-2.9 and vLLM-0.16 images are compatibility runtimes, not defaults for new tests. Confirm the expected
Torch and vLLM API before using a recorded launch command with them.

## Verification

Always set `--pwd /`; otherwise a host checkout named `vllm` can shadow the installed package. This probe uses
the same precedence as policy workers:

```bash
. .agents/ops/jupiter/production-runtime.env

apptainer exec --nv --pwd / --overlay "$POLICY_OVERLAY:ro" \
  --env "PYTHONPATH=$POLICY_PYDEPS" "$POLICY_SIF" \
  python -c 'import torch, torchtitan.distributed.expert_parallel as ep, vllm; from torchtitan.distributed.expert_parallel import expert_parallel; print(torch.__version__, vllm.__version__, ep.__file__, expert_parallel)'
```

## Multi-node fault contract

Run the pytest controller with a host Python that has pytest and the checkout available. The controller owns
the Slurm step. Only the remote node agents enter the SIF:

```bash
CHECKOUT=/e/scratch/jureap59/feuer1/codex/<clean-worktree>
. .agents/ops/jupiter/production-runtime.env
POLICY_PYTHONPATH="$CHECKOUT/skyrl-train:$POLICY_PYDEPS"

cd "$CHECKOUT/skyrl-train"
PYTHONPATH="$CHECKOUT/skyrl-train" "$HOST_PYTHON" -m pytest -s \
  tests/gpu/fault_injection/multi_node_nccl_contract.py \
  --confcutdir=tests/gpu/fault_injection \
  --node-agent-command-prefix="apptainer exec --nv --pwd / --overlay $POLICY_OVERLAY:ro --env PYTHONPATH=$POLICY_PYTHONPATH $POLICY_SIF"
```

Submit that command as the batch process of an allocation with exactly four four-GPU nodes. Do not wrap it in
`srun`; the controller launches and reaps the four node agents itself. The controller's Python is orchestration
only. Every distributed worker uses the image's `python` after the runtime prefix.
