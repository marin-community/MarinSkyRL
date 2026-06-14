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
#SBATCH --job-name=skyrl-stage3-dcp-0202sif
#SBATCH --output=/e/scratch/jureap59/feuer1/stage3_dcp_0202sif_%j.out

# ==============================================================================
# Stage 3 (vLLM DCP) — rollout correctness parity (DCP=N == DCP=1), the G2 gate,
# RE-RUN #2 on the CORRECT TARGET RUNTIME.
#
# The first two Stage-3 runs (jobs 856785 SIF, 856869 venv) BOTH ran on the
# torch-2.9 / vLLM-0.16-era stack (RL venv + skyrl_megatron_vllm.sif) and reported
# NO-GO. But DCP's real target is the torch-2.11 / vLLM-0.20.2rc0 SIF, which was
# NEVER tested. This runner pins the test to:
#     SIF = skyrl_megatron_vllm0202rc0_r3.sif  (vLLM 0.20.2rc0, torch 2.11.0+cu130)
# verified 2026-06-13: decode_context_parallel_size field present, get_dcp_group
# imports, vllm editable-installed at /opt/vllm_build/vllm/.
#
# CONFIRM-RUNTIME-FIRST: before the parity test, this job imports vllm+torch and
# HARD-ASSERTS torch starts with "2.11" and vllm pkg version == "0.20.2rc0".
# If it somehow lands on a 0.16 runtime again, it fails loud (exit 3) BEFORE any
# parity verdict — the whole point of run #2.
#
# Geometry (same as before for apples-to-apples): Qwen2.5-1.5B-Instruct, tp=4, dcp=2
# (heads=12/kv=2 -> q_per_kv=6; tp%dcp==0, tp>kv(2), dcp<=tp//kv=2, 6%2==0 -> valid).
# VLLM_ATTENTION_BACKEND=FLASH_ATTN pinned on BOTH dcp=1 and dcp=2 paths; the test
# logs the backend per engine and vLLM's own "Using <X> backend" line is in the log.
#
# NOTE: NOT torchrun — vLLM owns its own TP workers (mp executor); single-node
# in-process inside the SIF.
#
# IMPORTANT (env map §2d / reference_new_sif_torch211_multinode_fixes):
#   * vLLM lives at /opt/vllm_build/vllm/ in this SIF (editable .pth in
#     dist-packages, NOT PYTHONPATH-dependent). Do NOT add the R3 single-file
#     binds — the DCP test does not use R3 and binding R3 source over
#     /opt/vllm_build/vllm would shadow vllm -> ImportError.
#   * cwd MUST NOT contain a `vllm` subdir (e.g. /e/scratch/jureap59/feuer1 has a
#     `vllm` source clone that shadows the installed pkg). The test runs from
#     $WT/skyrl-train (no vllm subdir) -> safe.
#   * Single-node 4-GPU so the multi-node NCCL fixes likely aren't needed; if vLLM
#     init fails, retry with NCCL_P2P_DISABLE=1 (set DCP_RETRY_P2P_DISABLE=1).
# ==============================================================================
set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

# Worktree = dedicated MarinSkyRL clone on feuer/vllm-dcp (NOT the active RL src).
WT="${WT:-/e/scratch/jureap59/feuer1/vllm_dcp_wt}"
SIF="${SIF:-/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif}"
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
# Pin the attention backend IDENTICAL on both dcp=1 and dcp=2 paths.
# NOTE: on vLLM 0.20.2rc0 VLLM_ATTENTION_BACKEND is no longer a recognized env var
# (it logs "Unknown vLLM environment variable detected" and is IGNORED); the engine
# auto-selects the attention backend. It's the SAME (ignored) var on both dcp=1 and
# dcp=2, so there is no backend MISMATCH confound — both paths auto-select identically.
# Kept set for parity with the prior 0.16-era runs; the test logs the value seen.
export VLLM_ATTENTION_BACKEND="$ATTN"
# CRITICAL (0.20.2rc0 SIF gap): this SIF does NOT ship `flashinfer`, but vLLM
# 0.20.2rc0 defaults VLLM_USE_FLASHINFER_SAMPLER=True and TopKTopPSampler.__init__
# UNCONDITIONALLY does `from vllm.v1.attention.backends.flashinfer import
# FlashInferBackend` when that flag is on — crashing engine startup with
# `ModuleNotFoundError: No module named 'flashinfer'` BEFORE the graceful-fallback
# branch (observed: run #1 / job 857117, dcp=1 baseline never built). Setting this to
# 0 skips the import and uses the PyTorch-native/Triton sampler — which is also more
# deterministic and irrelevant to greedy (temp=0) parity. Set on BOTH paths.
export VLLM_USE_FLASHINFER_SAMPLER=0
# tp>1 standalone (NOT in a Ray actor) -> force vLLM TP workers to SPAWN, else they
# fork after the parent CUDA-inits and die "Cannot re-initialize CUDA in forked
# subprocess". (The test also sets these defensively.)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1

# Single-node 4-GPU: P2P generally fine. Opt-in retry lever if vLLM init fails.
P2P_ENV=""
if [ "${DCP_RETRY_P2P_DISABLE:-0}" = "1" ]; then
  P2P_ENV="--env NCCL_P2P_DISABLE=1"
  echo "[Stage3-DCP] NCCL_P2P_DISABLE=1 set (retry mode)"
fi

RUNLOG="/e/scratch/jureap59/feuer1/stage3_dcp_0202sif_${SLURM_JOB_ID}.log"
echo "=== Stage 3 DCP rollout-parity RE-RUN #2 (0.20.2rc0 SIF — correct runtime) ==="
echo "host: $(hostname)  WT=$WT  SIF=$SIF  MODEL=$MODEL  ATTN=$ATTN  RUNLOG=$RUNLOG"
git -C "$WT" log --oneline -1 || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$WT/skyrl-train"

# ---- CONFIRM-RUNTIME-FIRST: assert torch 2.11 / vllm 0.20.2rc0 (fail loud) ----
# Runs from $WT/skyrl-train (no vllm subdir) so the installed /opt/vllm_build/vllm
# is what imports. importlib.metadata is the reliable version source (vllm has no
# stable __version__ attr on this build per env map §0).
echo "=== [Stage3-DCP] RUNTIME ASSERTION (must be torch 2.11 / vllm 0.20.2rc0) ==="
srun apptainer exec --nv \
  --bind /e/scratch --bind /e/data1 \
  $P2P_ENV \
  "$SIF" \
  python -u -c "
import sys, torch
import importlib.metadata as m
tv = torch.__version__
vv = m.version('vllm')
print('[Stage3-DCP] ASSERTED torch =', tv)
print('[Stage3-DCP] ASSERTED vllm  =', vv)
import vllm
print('[Stage3-DCP] vllm file =', vllm.__file__)
from vllm.engine.arg_utils import EngineArgs
print('[Stage3-DCP] DCP field present:', 'decode_context_parallel_size' in EngineArgs.__dataclass_fields__)
from vllm.distributed.parallel_state import get_dcp_group
print('[Stage3-DCP] get_dcp_group import: OK')
ok = tv.startswith('2.11') and vv == '0.20.2rc0'
if not ok:
    print('[Stage3-DCP] FATAL: wrong runtime (expected torch 2.11 / vllm 0.20.2rc0). NOT a valid Stage-3 run.', file=sys.stderr)
    sys.exit(3)
print('[Stage3-DCP] RUNTIME OK -> proceeding to parity test.')
" 2>&1 | tee "$RUNLOG"
ASSERT_RC=${PIPESTATUS[0]}
if [ "$ASSERT_RC" -ne 0 ]; then
  echo "=== Stage 3 DCP RUNTIME ASSERTION FAILED (rc=$ASSERT_RC) — aborting, NOT a valid verdict ==="
  exit "$ASSERT_RC"
fi

# ---- The parity test (only reached if runtime is the confirmed 0.20.2rc0 target) ----
echo "=== [Stage3-DCP] RUNNING ROLLOUT-PARITY TEST ==="
srun apptainer exec --nv \
  --bind /e/scratch --bind /e/data1 \
  --env PYTHONPATH="$WT/skyrl-train" \
  --env LIBRARY_PATH=/.singularity.d/libs \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env DCP_PARITY_MODEL="$MODEL" \
  --env DCP_PARITY_TP=4 --env DCP_PARITY_DCP=2 \
  --env VLLM_ATTENTION_BACKEND="$ATTN" \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ENABLE_V1_MULTIPROCESSING=1 \
  $P2P_ENV \
  "$SIF" \
  python -u tests/gpu/test_dcp_rollout_parity.py 2>&1 | tee -a "$RUNLOG"
echo "=== Stage 3 DCP 0.20.2rc0-SIF parity EXIT ${PIPESTATUS[0]} ==="
