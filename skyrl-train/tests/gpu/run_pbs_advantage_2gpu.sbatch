#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --reservation=reformo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:2
#SBATCH --time=00:25:00
#SBATCH --job-name=pbs-adv-2gpu
#SBATCH --output=/e/scratch/jureap59/feuer1/tmp_pbs_2gpu/logs/%x_%j.out

# Stage C (F6) 2-GPU gate: rloo_n_pbs advantage dispatcher on CUDA + the real
# 2-GPU ppo_train policy-loss path (loss-value parity / no denominator break).
# Runs in the shared envs/rl venv (torch 2.9 + fsdp2 — what Stage B validated on).
set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

export DCFT=/e/scratch/jureap59/feuer1/OpenThoughts-Agent
WORKDIR="$DCFT"
SKYRL="$DCFT/SkyRL/skyrl-train"
PY="$DCFT/envs/rl/bin/python"
cd "$SKYRL"

module load GCC/14.3.0 nvidia-compilers/25.9-CUDA-13 2>/dev/null || true
if [ -f "$DCFT/hpc/dotenv/jupiter.env" ]; then
  set -a; source "$DCFT/hpc/dotenv/jupiter.env"; set +a
fi

export GLOO_USE_IPV6=0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib0
export VLLM_FORCE_IPV4=1
export VLLM_USE_V1=1
export UV_USE_IO_URING=0
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=offline
# Compute nodes have no internet; Qwen2.5-0.5B-Instruct is pre-cached in
# $HF_HUB_CACHE (set by jupiter.env) -> force offline to use the cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export PYTHONPATH="$SKYRL:$PYTHONPATH"

echo "=== PBS 2-GPU gate START $(date) on $(hostname) ==="
nvidia-smi -L || true

"$PY" -m pytest -s -vvv \
  tests/gpu/gpu_ci/test_pbs_advantage_2gpu.py \
  2>&1
RC=$?
echo "=== PBS 2-GPU gate DONE rc=$RC $(date) ==="
exit $RC
