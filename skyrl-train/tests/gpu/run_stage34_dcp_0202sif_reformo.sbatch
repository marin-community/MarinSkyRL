#!/bin/bash
#SBATCH --account=reformo
#SBATCH --reservation=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --job-name=skyrl-stage34-dcp-0202sif
#SBATCH --output=/e/scratch/jureap59/feuer1/stage34_dcp_0202sif_%j.out

# ==============================================================================
# Stage 3 (RE-GATED to bf16-sharded-attention tolerance) + Stage 4 (long-ctx KV
# OOM->OK ship gate), vLLM DCP, on the CORRECT target runtime:
#     SIF = skyrl_megatron_vllm0202rc0_r3.sif  (vLLM 0.20.2rc0, torch 2.11.0+cu130)
#
# WHY re-gated: the vLLM-fix investigation (jobs 856785/856869/857121) concluded
# there is NO DCP bug. The free-running greedy divergence is bf16 floating-point
# non-associativity of the split-KV sharded-attention recombine (ag_rs) — a per-step
# ~3e-2 residual that accumulates and flips NEAR-TIE argmaxes over decode. The combine
# is correct within bf16; exact greedy reproducibility is neither achievable nor
# required (TIS corrects logprob mismatches far larger than this). So Stage 3 is now a
# TOLERANCE gate (teacher-forced p50/p99/max |Δlogprob| within bf16 noise + high-but-
# not-100% argmax agreement + all flips are near-ties). No vLLM change needed.
#
# Geometry: Qwen2.5-1.5B-Instruct, tp=4, dcp=2 (heads=12/kv=2 -> q_per_kv=6; valid).
# NOT torchrun — vLLM owns its own TP workers (single-node in-process inside the SIF).
#
# Runtime gotchas (env map / reference_new_sif_torch211_multinode_fixes):
#   * vLLM lives at /opt/vllm_build/vllm/ (editable .pth). Do NOT bind R3 source.
#   * cwd MUST NOT contain a `vllm` subdir (would shadow the install) -> run from
#     $WT/skyrl-train.
#   * VLLM_USE_FLASHINFER_SAMPLER=0 (SIF ships no flashinfer; default-True crashes
#     TopKTopPSampler at startup) — set on every engine path.
#   * LIBRARY_PATH=/.singularity.d/libs (Triton link gotcha on this SIF).
# ==============================================================================
set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

WT="${WT:-/e/scratch/jureap59/feuer1/vllm_dcp_wt}"
SIF="${SIF:-/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif}"
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

RUNLOG="/e/scratch/jureap59/feuer1/stage34_dcp_0202sif_${SLURM_JOB_ID}.log"
echo "=== Stage 3 (re-gated) + Stage 4 (OOM->OK) DCP — 0.20.2rc0 SIF ==="
echo "host: $(hostname)  WT=$WT  SIF=$SIF  MODEL=$MODEL  RUNLOG=$RUNLOG"
git -C "$WT" log --oneline -1 || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$WT/skyrl-train"

# ---- CONFIRM-RUNTIME-FIRST: assert torch 2.11 / vllm 0.20.2rc0 (fail loud) ----
echo "=== [DCP] RUNTIME ASSERTION (torch 2.11 / vllm 0.20.2rc0) ==="
srun apptainer exec --nv \
  --bind /e/scratch --bind /e/data1 \
  "$SIF" \
  python -u -c "
import sys, torch
import importlib.metadata as m
tv = torch.__version__; vv = m.version('vllm')
print('[DCP] ASSERTED torch =', tv); print('[DCP] ASSERTED vllm  =', vv)
import vllm; print('[DCP] vllm file =', vllm.__file__)
from vllm.engine.arg_utils import EngineArgs
print('[DCP] DCP field present:', 'decode_context_parallel_size' in EngineArgs.__dataclass_fields__)
from vllm.distributed.parallel_state import get_dcp_group; print('[DCP] get_dcp_group: OK')
ok = tv.startswith('2.11') and vv == '0.20.2rc0'
sys.exit(0 if ok else 3)
" 2>&1 | tee "$RUNLOG"
ASSERT_RC=${PIPESTATUS[0]}
if [ "$ASSERT_RC" -ne 0 ]; then
  echo "=== RUNTIME ASSERTION FAILED (rc=$ASSERT_RC) — aborting ==="
  exit "$ASSERT_RC"
fi

COMMON_ENV=(
  --env PYTHONPATH="$WT/skyrl-train"
  --env LIBRARY_PATH=/.singularity.d/libs
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env DCP_PARITY_MODEL="$MODEL"
  --env DCP_PARITY_TP=4 --env DCP_PARITY_DCP=2
  --env VLLM_USE_FLASHINFER_SAMPLER=0
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn
  --env VLLM_ENABLE_V1_MULTIPROCESSING=1
)

# ---- STAGE 3: re-gated rollout-parity tolerance gate ----
echo "=== [DCP] STAGE 3 — RE-GATED rollout-parity tolerance gate ==="
srun apptainer exec --nv \
  --bind /e/scratch --bind /e/data1 \
  "${COMMON_ENV[@]}" \
  "$SIF" \
  python -u tests/gpu/test_dcp_rollout_parity.py 2>&1 | tee -a "$RUNLOG"
S3_RC=${PIPESTATUS[0]}
echo "=== [DCP] STAGE 3 EXIT ${S3_RC} ==="

# ---- STAGE 4: long-ctx KV OOM->OK ship gate ----
# Qwen2.5-1.5B caps max_position_embeddings at 32768, and at tp=4 the tiny KV fits
# trivially below that cap -> a crisp single-GPU OOM-at-32k is NOT achievable. The
# scope's endorsed path for that case is the MEASURED KV-HEADROOM gate: with
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (lifts the position-embedding cap so KV becomes the
# BINDING constraint; Stage 4 only tests whether the KV cache INITIALIZES, not output
# quality, so the RoPE-NaN-beyond-32k caveat is irrelevant here), at a low gpu_util we
# ladder the max_model_len and record the largest that initializes at dcp=1 vs dcp=2.
# DCP shards the decode KV along the token dim, removing the tp>kv_heads (4>2)
# duplication, so dcp=2's per-GPU KV/token is ~half dcp=1's -> dcp=2 reaches a strictly
# LARGER max context. That ratio (~2x) is the DCP long-context win, independent of the
# absolute available-KV. (oom-mode is still wired + auto-falls-back here.)
echo "=== [DCP] STAGE 4 — long-ctx KV headroom (OOM->OK) ship gate ==="
srun apptainer exec --nv \
  --bind /e/scratch --bind /e/data1 \
  "${COMMON_ENV[@]}" \
  --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --env DCP_OOM_MODE="${DCP_OOM_MODE:-headroom}" \
  --env DCP_HEADROOM_GPUUTIL="${DCP_HEADROOM_GPUUTIL:-0.20}" \
  --env DCP_HEADROOM_LADDER="${DCP_HEADROOM_LADDER:-65536,131072,196608,262144,393216,524288}" \
  --env DCP_OOM_MAXLEN="${DCP_OOM_MAXLEN:-262144}" \
  --env DCP_OOM_GPUUTIL="${DCP_OOM_GPUUTIL:-0.20}" \
  --env DCP_OOM_PROMPT_LEN="${DCP_OOM_PROMPT_LEN:-131072}" \
  --env DCP_OOM_GEN_LEN="${DCP_OOM_GEN_LEN:-256}" \
  "$SIF" \
  python -u tests/gpu/test_dcp_longctx_oom.py 2>&1 | tee -a "$RUNLOG"
S4_RC=${PIPESTATUS[0]}
echo "=== [DCP] STAGE 4 EXIT ${S4_RC} ==="

echo "=== [DCP] SUMMARY: Stage3 rc=${S3_RC}  Stage4 rc=${S4_RC} ==="
[ "$S3_RC" -eq 0 ] && [ "$S4_RC" -eq 0 ] && echo "=== [DCP] BOTH STAGES PASS ===" || echo "=== [DCP] AT LEAST ONE STAGE FAILED ==="
exit 0
