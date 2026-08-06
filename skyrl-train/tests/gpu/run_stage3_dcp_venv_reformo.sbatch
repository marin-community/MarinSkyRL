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
#SBATCH --job-name=skyrl-stage3-dcp-venv
#SBATCH --output=/e/scratch/jureap59/feuer1/stage3_dcp_venv_%j.out

# ==============================================================================
# Stage 3 (vLLM DCP) — rollout-parity RE-RUN to isolate the original NO-GO confound.
#
# Two changes vs run_stage3_dcp_reformo.sbatch (the SIF runner that gave NO-GO):
#   (1) RUNTIME = the Jupiter standard-RL VENV vLLM ($WORKDIR/envs/rl), run
#       NATIVELY (no apptainer/SIF). This is the runtime the real RL rollout path
#       uses. (Note: this venv vLLM is the mlfoundations/vllm fork; its declared
#       version is 0.16.0 / runtime __version__='dev' — same fork family as the SIF;
#       the DCP attention code (flash_attn.py) is byte-identical between the two.)
#   (2) ATTENTION BACKEND PINNED IDENTICAL on BOTH dcp=1 and dcp=2 paths via
#       VLLM_ATTENTION_BACKEND=FLASH_ATTN (the DCP-capable GQA backend), so a
#       backend AUTO-SELECT mismatch between the two paths cannot confound parity.
#
# The test prints the pinned backend per engine; vLLM's own "Using <X> backend"
# startup line is in the run log too -> confirm both paths used FLASH_ATTN.
#
# Same model/geometry as the SIF run: Qwen2.5-1.5B-Instruct, tp=4, dcp=2
# (heads=12/kv=2 -> q_per_kv=6; tp%dcp==0, tp>kv, dcp<=tp//kv=2, 6%2==0 -> valid).
# NOT torchrun — vLLM owns its TP workers (mp executor); single-node in-process.
# ==============================================================================
set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

WT="${WT:-/e/scratch/jureap59/feuer1/vllm_dcp_wt}"
VENV="${VENV:-/e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl}"
PY="$VENV/bin/python"
MODEL="${DCP_PARITY_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ATTN="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

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
# ISOLATION (2): pin the attention backend IDENTICAL on both dcp=1 and dcp=2.
export VLLM_ATTENTION_BACKEND="$ATTN"
# tp>1 standalone (NOT in a Ray actor) -> force vLLM TP workers to SPAWN.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
# Run the worktree's skyrl-train (Stage-1/2 DCP plumbing under test) first on path,
# but the test imports only vllm/transformers so this is belt-and-suspenders.
export PYTHONPATH="$WT/skyrl-train:${PYTHONPATH:-}"

RUNLOG="/e/scratch/jureap59/feuer1/stage3_dcp_venv_${SLURM_JOB_ID}.log"
echo "=== Stage 3 DCP rollout-parity VENV re-run (backend-pinned) ==="
echo "host: $(hostname)  WT=$WT  VENV=$VENV  MODEL=$MODEL  ATTN=$ATTN  RUNLOG=$RUNLOG"
git -C "$WT" log --oneline -1 || true
"$PY" -c "import vllm; print('vllm.__version__ =', vllm.__version__)" 2>/dev/null || true
"$PY" -m pip show vllm 2>/dev/null | grep -iE '^(Name|Version|Location)' || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$WT/skyrl-train"
srun "$PY" -u tests/gpu/test_dcp_rollout_parity.py 2>&1 | tee "$RUNLOG"
echo "=== Stage 3 DCP venv parity EXIT ${PIPESTATUS[0]} ==="
