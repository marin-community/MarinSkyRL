#!/bin/bash
# Rebuild the GatedDeltaNet kernel overlay (tilelang 0.1.8 + FlashQLA fused GDN
# fwd+bwd) against the NATIVE torch-2.11 SIF. Run on a GPU (reformo) node so the
# aarch64 arch + nvcc are visible; FlashQLA/tilelang are wheels/light builds.
#
# Mirrors the Stage-8 Path-B recipe (notes/skyrl/stage8_scope.md) but targets the
# new torch-2.11 SIF. NO TransformerEngine (FSDP2). Installs --no-deps so the
# pinned torch 2.11 / triton / vLLM 0.20.2rc0 stack is NOT perturbed.
set -x
B=/e/scratch/jureap59/feuer1
SIF=$B/containers/skyrl_megatron_vllm_r3_torch211.sif
OV=$B/containers/fla_tilelang_torch211_overlay.img
LOG=$B/build/fla_overlay_torch211.log
exec > "$LOG" 2>&1
echo "=== START $(date) ==="
module load Apptainer 2>/dev/null || true

if [ ! -f "$SIF" ]; then echo "FATAL: torch211 SIF not built yet: $SIF"; exit 2; fi

# 1. fresh 4 GiB writable ext3 overlay (do NOT touch the torch-2.9 fla overlay)
rm -f "$OV"
apptainer overlay create --size 4096 "$OV"
echo "OVERLAY_CREATED $(date)"

# 2. install INTO the overlay (writable). --no-deps where a dep would touch the
#    pinned stack. tilelang pulls z3-solver + apache-tvm-ffi (keep at the SIF's
#    version; FlashQLA pins 0.1.9 but install --no-deps to avoid a downgrade fight).
apptainer exec --nv --writable --overlay "$OV" --bind /e/scratch "$SIF" bash -lc '
  set -e
  unset CMAKE_PREFIX_PATH PKG_CONFIG_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
  unset EBROOTGCCCORE EBROOTBINUTILS EBROOTZLIB EBROOTMAKE GCC_HOME
  export CC=/usr/bin/gcc CXX=/usr/bin/g++
  export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
  export TORCH_CUDA_ARCH_LIST="9.0+PTX"
  echo "=== pre-install pinned-stack snapshot ==="
  python -c "import torch,sys;print(\"py\",sys.version.split()[0]);print(\"torch\",torch.__version__,torch.version.cuda)"
  python -c "import triton;print(\"triton\",triton.__version__)" 2>&1 | tail -1
  python -c "import vllm;print(\"vllm\",vllm.__version__)" 2>&1 | tail -1
  python -c "import apache_tvm_ffi as t;print(\"tvm_ffi\",t.__version__)" 2>&1 | tail -1 || true

  echo "=== install tilelang 0.1.8 (aarch64 abi3 wheel; pulls z3-solver) ==="
  python -m pip install --no-cache-dir --no-build-isolation tilelang==0.1.8
  echo "=== install flash-linear-attention 0.5.0 --no-deps (masked at runtime; FlashQLA is self-contained) ==="
  python -m pip install --no-cache-dir --no-deps flash-linear-attention==0.5.0
  echo "=== install FlashQLA --no-deps (git) ==="
  rm -rf /tmp/FlashQLA
  git clone https://github.com/QwenLM/FlashQLA /tmp/FlashQLA
  ( cd /tmp/FlashQLA && git checkout 6ef4858 2>/dev/null || true )
  python -m pip install --no-cache-dir --no-deps -v /tmp/FlashQLA

  echo "=== verify FlashQLA imports + tilelang loads under torch 2.11 ==="
  python -c "import tilelang; print(\"TILELANG_OK\", getattr(tilelang,\"__version__\",\"?\"))"
  python -c "import flash_qla; print(\"FLASHQLA_OK\", flash_qla.__version__, \"has chunk:\", hasattr(flash_qla,\"chunk_gated_delta_rule\"))"
  python -c "from flash_qla import chunk_gated_delta_rule_fwd, chunk_gated_delta_rule_bwd; print(\"FLASHQLA_FUNCTIONAL_OK\")"
  echo "=== POST-install pinned-stack snapshot (must be byte-identical) ==="
  python -c "import torch;print(\"torch\",torch.__version__,torch.version.cuda)"
  python -c "import triton;print(\"triton\",triton.__version__)" 2>&1 | tail -1
  python -c "import vllm;print(\"vllm\",vllm.__version__)" 2>&1 | tail -1
'
echo "FLA_OVERLAY_TORCH211_BUILD_RC=$?"
ls -la "$OV" 2>&1
echo "=== FLA_OVERLAY_TORCH211_DONE $(date) ==="
