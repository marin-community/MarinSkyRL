#!/bin/bash
# Jupiter-only SIF test: four-node EP4/FSDP4 MoE dispatch-stage contract.
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=moe-dispatch

set -euo pipefail

CHECKOUT=${CHECKOUT:?set CHECKOUT to the staged MarinSkyRL checkout}
ARTIFACT_ROOT=${ARTIFACT_ROOT:?set ARTIFACT_ROOT to a bounded shared output directory}
. "$CHECKOUT/.agents/ops/jupiter/production-runtime.env"

mkdir -p "$ARTIFACT_ROOT"
POLICY_PYTHONPATH="$CHECKOUT:$CHECKOUT/skyrl-train:$CHECKOUT/skyrl-gym:$POLICY_PYDEPS"
NODE_PREFIX="apptainer exec --nv --pwd / --overlay $POLICY_OVERLAY:ro --env PYTHONPATH=$POLICY_PYTHONPATH $POLICY_SIF"

cd "$CHECKOUT/skyrl-train"
PYTHONPATH="$CHECKOUT:$CHECKOUT/skyrl-train" "$HOST_PYTHON" -m pytest -s \
  tests/gpu/fault_injection/multi_node_moe_dispatch_contract.py \
  --confcutdir=tests/gpu/fault_injection \
  --node-agent-command-prefix="$NODE_PREFIX" \
  --debug-artifact-root="$ARTIFACT_ROOT"
