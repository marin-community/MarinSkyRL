#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=skyrl-stage6-e2e
#SBATCH --output=/e/data1/datasets/playground/ot-baf/experiments/_ray_logs/%x_%j.out

# Stage 6 e2e: small-MoE RL step with backprop + router replay ON.
# Lean, self-contained runner (no Ray multi-node orchestration) mirroring the
# Jupiter RL preamble from hpc/sbatch_rl/universal_rl.sbatch (modules + jupiter.env
# + envs/rl venv + the Jupiter IPv4/NCCL/uvloop env vars).
set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

export DCFT=/e/scratch/jureap59/feuer1/OpenThoughts-Agent
WORKDIR="$DCFT"
cd "$WORKDIR"

# --- Modules (Jupiter HPC config: GCC 14 + CUDA 13) ---
module load GCC/14.3.0 nvidia-compilers/25.9-CUDA-13 2>/dev/null || true

# --- Source the Jupiter dotenv (HF cache, LD_LIBRARY_PATH, SKYRL_HOME, etc.) ---
if [ -f "$DCFT/hpc/dotenv/jupiter.env" ]; then
  set -a
  source "$DCFT/hpc/dotenv/jupiter.env"
  set +a
fi

# --- RL venv ---
RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"
source "$RL_ENV_DIR/bin/activate"

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# --- Jupiter networking / stability env (subset of hpc.py jupiter env_vars + universal_rl) ---
export GLOO_USE_IPV6=0
export NCCL_SOCKET_FAMILY=AF_INET
export VLLM_FORCE_IPV4=1
export VLLM_SKIP_FLAG_DISCOVERY=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_V1=1
export UV_USE_IO_URING=0
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export DISABLE_AIOHTTP_TRANSPORT=True

# Live SkyRL source on the path (run the just-pulled test + skyrl_train).
export PYTHONPATH="$SKYRL_HOME/skyrl-train:$WORKDIR:${PYTHONPATH:-}"

# All 4 GPUs visible to this single-node EP=2 x FSDP=2 + EP=2-inference test.
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=== Stage 6 e2e launch ==="
echo "host: $(hostname)  python: $(which python)"
python --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "SKYRL_HOME=$SKYRL_HOME"
git -C "$SKYRL_HOME" log --oneline -1 || true

cd "$SKYRL_HOME/skyrl-train"
srun --export=ALL python tests/gpu/test_e2e_moe_rl_step.py
echo "=== Stage 6 e2e EXIT $? ==="
