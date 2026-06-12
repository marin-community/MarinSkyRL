#!/bin/bash
#SBATCH --account=reformo
#SBATCH --reservation=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=skyrl-stage6-cp-e2e
#SBATCH --output=/e/data1/datasets/playground/ot-baf/experiments/_ray_logs/%x_%j.out

# ---------------------------------------------------------------------------
# Stage 6 (FSDP2 CP) — E2E GRPO parity + long-context OOM->OK + CP+EP.
#
# Runs the SELF-CONTAINED torchrun e2e test (NOT the Ray PPO trainer). With 4
# GPUs it exercises TEST 1 (e2e GRPO parity), TEST 2 (OOM->OK), TEST 4 (resume),
# and TEST 3 (CP+EP 4-D mesh, cp2 x ep2). With 2 GPUs it runs TEST 1/2/4 (dense
# CP) and SKIPs TEST 3.
#
# NOTE: distinct from run_stage6_reformo.sbatch (that one is the R3/MoE stage-6
# = test_e2e_moe_rl_step.py, a different plan). This is the CONTEXT-PARALLEL
# stage 6.
#
# The SIF bakes SkyRL at /opt/SkyRL which shadows the worktree clone — we set
# PYTHONPATH to the worktree's skyrl-train so the model_wrapper/cp_utils UNDER
# TEST are imported (the test asserts this); LIBRARY_PATH for Triton JIT gcc.
# ---------------------------------------------------------------------------
set -eo pipefail
ulimit -c 0

# Worktree (dedicated MarinSkyRL clone on feuer/fsdp2-cp; NOT the active RL src).
WT="${WT:-/e/scratch/jureap59/feuer1/cp_stage2_wt}"
SIF="${SIF:-/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif}"
NPROC="${NPROC:-4}"   # 4 = CP+EP; set NPROC=2 for dense-CP-only.

export GLOO_USE_IPV6=0
export NCCL_SOCKET_FAMILY=AF_INET
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SKYRL_CP_REQUIRE_RIGHT_ALIGN=1

RUNLOG="/e/scratch/jureap59/feuer1/stage6_cp_e2e_${SLURM_JOB_ID}.log"
echo "=== Stage 6 CP e2e launch (nproc=${NPROC}) ==="
echo "host: $(hostname)  WT=$WT  SIF=$SIF  RUNLOG=$RUNLOG"
git -C "$WT" log --oneline -1 || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$WT/skyrl-train"
srun apptainer exec --nv \
  --env PYTHONPATH="$WT/skyrl-train" \
  --env LIBRARY_PATH=/.singularity.d/libs \
  --env SKYRL_CP_REQUIRE_RIGHT_ALIGN=1 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  "$SIF" \
  torchrun --nproc-per-node="${NPROC}" tests/gpu/test_cp_e2e_grpo.py 2>&1 | tee "$RUNLOG"
echo "=== Stage 6 CP e2e EXIT ${PIPESTATUS[0]} ==="
