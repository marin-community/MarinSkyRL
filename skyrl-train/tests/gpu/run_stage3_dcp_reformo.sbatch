#!/bin/bash
#SBATCH --account=reformo
#SBATCH --reservation=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=skyrl-stage3-dcp-parity
#SBATCH --output=/e/scratch/jureap59/feuer1/stage3_dcp_parity_%j.out

# ==============================================================================
# Stage 3 (vLLM DCP) — rollout correctness parity (DCP=N == DCP=1), the G2 gate.
#
# Builds TWO vllm.LLM engines on the SAME tiny GQA model (Qwen2.5-1.5B-Instruct:
# heads=12, kv_heads=2 -> q_per_kv=6), SAME seed, enforce_eager, on ONE node /
# 4 GH200:
#   Engine A: tp=4, dcp=1     Engine B: tp=4, dcp=2
# (geometry: tp%dcp==0, tp>kv_heads(2), dcp<=tp//kv(=2), q_per_kv(6)%dcp==0 -> valid)
# and asserts: greedy token_ids IDENTICAL (G2 make-or-break) + logprobs allclose
# (atol 1e-2, absorbing DCP's AllGather+ReduceScatter fp reduction-order shift).
# It extracts token_ids/logprobs with the EXACT _postprocess_outputs logic the
# TIS path consumes.
#
# NOTE: NOT torchrun — vLLM owns its own TP worker processes (tp=4 via the in-
# engine mp/uniproc executor). The batch script runs the test in-process on the
# single node, inside the SIF.
#
# The SIF bakes SkyRL at /opt/SkyRL which shadows a worktree clone; PYTHONPATH
# points at the worktree's skyrl-train so the Stage-1/2 DCP plumbing UNDER TEST
# is what is imported. The SIF's vLLM exposes DCP (Stage-1 verified: SIF vllm
# 0.16.0 has decode_context_parallel_size + get_dcp_group).
# ==============================================================================
set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

# Worktree = dedicated MarinSkyRL clone on feuer/vllm-dcp (NOT the active RL src).
WT="${WT:-/e/scratch/jureap59/feuer1/vllm_dcp_wt}"
SIF="${SIF:-/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm.sif}"
MODEL="${DCP_PARITY_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

export DCFT=/e/scratch/jureap59/feuer1/OpenThoughts-Agent
WORKDIR="$DCFT"
cd "$WORKDIR"
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
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DCP_PARITY_MODEL="$MODEL"
export DCP_PARITY_TP=4
export DCP_PARITY_DCP=2
# tp>1 standalone (NOT in a Ray actor) -> force vLLM TP workers to SPAWN, else
# they fork after the parent CUDA-inits and die "Cannot re-initialize CUDA in
# forked subprocess". (The test also sets these defensively.)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1

RUNLOG="/e/scratch/jureap59/feuer1/stage3_dcp_parity_${SLURM_JOB_ID}.log"
echo "=== Stage 3 DCP rollout-parity launch ==="
echo "host: $(hostname)  WT=$WT  SIF=$SIF  MODEL=$MODEL  RUNLOG=$RUNLOG"
git -C "$WT" log --oneline -1 || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$WT/skyrl-train"
srun apptainer exec --nv \
  --bind /e/scratch --bind /e/data1 \
  --env PYTHONPATH="$WT/skyrl-train" \
  --env LIBRARY_PATH=/.singularity.d/libs \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env DCP_PARITY_MODEL="$MODEL" \
  --env DCP_PARITY_TP=4 --env DCP_PARITY_DCP=2 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ENABLE_V1_MULTIPROCESSING=1 \
  "$SIF" \
  python -u tests/gpu/test_dcp_rollout_parity.py 2>&1 | tee "$RUNLOG"
echo "=== Stage 3 DCP parity EXIT ${PIPESTATUS[0]} ==="
