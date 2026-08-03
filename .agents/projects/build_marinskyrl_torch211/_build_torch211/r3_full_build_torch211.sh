#!/bin/bash
# End-to-end NATIVE torch-2.11 r3 SIF build. Self-contained; run detached in tmux
# on login03. Do NOT interrupt. Expect MANY hours given GPFS small-file slowness
# (FlashMLA / cutlass) + the NGC 26.02 base pull on the first run.
set -x
B=/e/scratch/jureap59/feuer1/build
LOG=$B/r3_full_build_torch211.log
exec > "$LOG" 2>&1
echo "=== START $(date) ==="
module load Apptainer 2>/dev/null || true
which apptainer

echo "=== STAGE 0: ensure flashmla_prebuilt present (reuse pre-clone) ==="
if [ -d "$B/flashmla_prebuilt/FlashMLA/csrc/cutlass/include" ]; then
  echo "FLASHMLA_PREBUILT_REUSE_OK"
else
  echo "FATAL: flashmla_prebuilt missing -- pre-clone first (see r3_full_build.sh STAGE A)"; exit 3
fi

echo "=== STAGE B-pre: clear stale host-configured CMake trees in vllm_fork/.deps ==="
# *-subbuild/*-build CMakeCache.txt bake the HOST configure path; the container
# binds the build dir at /mnt -> CMake aborts on cachefile-dir mismatch. Drop the
# configured trees so CMake reconfigures fresh at /mnt INSIDE the container; KEEP
# the *-src clones (slow GPFS checkouts: cutlass/deepgemm/triton_kernels-src).
cd "$B"
if [ -d vllm_fork/.deps ]; then
  for d in vllm_fork/.deps/*-subbuild vllm_fork/.deps/*-build; do
    [ -d "$d" ] || continue
    echo "clearing stale CMake tree: $d ($(date))"
    rm -rf "$d"
  done
fi
rm -rf vllm_fork/.deps/flashmla-src vllm_fork/.deps/flashmla-build vllm_fork/.deps/flashmla-subbuild
echo "STAGE_B_PRE_CLEAR_DONE $(date)"
ls -la vllm_fork/.deps/ 2>&1

echo "=== STAGE B: apptainer build of NATIVE torch-2.11 r3 SIF (build dir bound at /mnt) ==="
cd "$B"
# Bootstrap pulls nvcr.io/nvidia/pytorch:26.02-py3 (anonymous; multi-arch -> aarch64/sbsa)
apptainer build --fakeroot \
  --bind "$B:/mnt" \
  /e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm_r3_torch211.sif \
  "$B/skyrl_megatron_vllm_r3_torch211.def"
echo "APPTAINER_BUILD_RC=$?"
ls -la /e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm_r3_torch211.sif 2>&1
echo "=== R3_TORCH211_FULL_BUILD_DONE $(date) ==="
