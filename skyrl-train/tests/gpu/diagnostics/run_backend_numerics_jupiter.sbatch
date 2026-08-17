#!/bin/bash
# Jupiter-only SIF test: this manual Slurm job uses the standard Jupiter RL environment.
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=backend-numerics

set -euo pipefail

CHECKOUT=${CHECKOUT:?set CHECKOUT to the staged MarinSkyRL checkout}
ARTIFACT_ROOT=${ARTIFACT_ROOT:?set ARTIFACT_ROOT to a bounded shared output directory}
CHECKOUT_REVISION=${CHECKOUT_REVISION:?set CHECKOUT_REVISION before entering the compute node}
FROZEN_PYTHON=${FROZEN_PYTHON:?set FROZEN_PYTHON to the frozen Megatron environment}

mkdir -p "$ARTIFACT_ROOT"
export SKYRL_NUMERICS_ARTIFACT_DIR="$ARTIFACT_ROOT/artifacts"
export TRITON_CACHE_DIR="$ARTIFACT_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$ARTIFACT_ROOT/torchinductor-cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export PYTHONPATH="$CHECKOUT:$CHECKOUT/skyrl-train"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_USE_IPV6=0
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=127.0.0.1
FROZEN_CUDA_ACTIVATION="$ARTIFACT_ROOT/frozen-cuda-runtime.env"
"$FROZEN_PYTHON" -m cloud.iris.env_vars write-frozen-cuda-runtime "$FROZEN_CUDA_ACTIVATION"
. "$FROZEN_CUDA_ACTIVATION"
cd "$CHECKOUT/skyrl-train"

run_in_policy_runtime() {
  if [[ $1 != python ]]; then
    echo "run_in_policy_runtime expects a Python command, got $1" >&2
    return 2
  fi
  shift
  "$FROZEN_PYTHON" "$@"
}

{
  echo "$CHECKOUT_REVISION"
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
  run_in_policy_runtime python -c 'import megatron.core, torch, torchtitan, transformers; print(torch.__version__, torch.version.cuda, megatron.core.__file__, torchtitan.__file__, transformers.__version__)'
} > "$ARTIFACT_ROOT/runtime.txt" 2>&1

declare -a RESULTS=()

run_diagnostic() {
  local name=$1
  shift
  set +e
  run_in_policy_runtime "$@" 2>&1 | tee "$ARTIFACT_ROOT/$name.log"
  local status=${PIPESTATUS[0]}
  set -e
  RESULTS+=("$name=$status")
}

run_diagnostic t1 python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29601 --nproc-per-node=4 \
  -m pytest -s \
  tests/gpu/diagnostics/backend_numerics_t1_gradient_norms.py
run_diagnostic t2-fsdp python -m pytest -s \
  tests/gpu/diagnostics/backend_numerics_t2_logprob_gradients.py -k fsdp
run_diagnostic t2-megatron-tp1 python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29602 \
  --nproc-per-node=1 -m pytest -s \
  tests/gpu/diagnostics/backend_numerics_t2_logprob_gradients.py -k megatron
run_diagnostic t2-megatron-tp2 python -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29603 \
  --nproc-per-node=2 -m pytest -s \
  tests/gpu/diagnostics/backend_numerics_t2_logprob_gradients.py -k 'megatron and not inference_only'
run_diagnostic t3 python -m pytest -s \
  tests/gpu/diagnostics/backend_numerics_t3_moe_attention_gradients.py

printf '%s\n' "${RESULTS[@]}" | tee "$ARTIFACT_ROOT/summary.txt"
