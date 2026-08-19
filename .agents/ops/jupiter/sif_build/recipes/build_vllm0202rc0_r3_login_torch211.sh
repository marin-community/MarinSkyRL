#!/bin/bash
# =============================================================================
# build_vllm0202rc0_r3_login_torch211.sh   (LOGIN-NODE build, NOT sbatch)  PATH B
#
# Reproduces the vLLM-0.20.2rc0 fork + native R3 routed-experts SIF, building the
# vLLM COMPILE on the Jupiter LOGIN node (direct internet) instead of an offline
# booster compute node. The prior 5 sbatch attempts (latest 850017) all died at
# vLLM CMake FetchContent `git clone github.com/nvidia/cutlass.git` -- offline node
# cannot reach github:443. Login nodes have internet -> cutlass clone succeeds.
#
# PATH A (keep in-SIF torch 2.9) was ATTEMPTED FIRST and got PAST cutlass (used
# VLLM_CUTLASS_SRC_DIR, compiled to [27/394]) but then hit a HARD source-level
# block: csrc/cutlass_extensions/common.hpp #includes
# <torch/headeronly/util/shim_utils.h>, which exists in torch 2.11 but NOT in the
# in-SIF NGC torch 2.9. The fork's C++ extensions GENUINELY require torch 2.11
# headers -- not just a strippable pin. So Path A is impossible and we MUST bump.
#
# STRATEGY (Path B -- the user's PROVEN recipe: bump torch to 2.11):
#   base SIF = skyrl_megatron.sif (NGC 25.09: torch 2.9, TE 2.7, flash_attn 2.7.4,
#              transformers 5.10.1, Megatron, apex, SkyRL editable).
#   + apptainer build --sandbox on the login node (staged on login-local /tmp;
#     exa_scratch is near its inode hard-limit).
#   + apptainer exec --writable --nv INSIDE the sandbox on the login node:
#       1. BUMP torch -> 2.11.0+cu130 (+ torchvision/torchaudio) from the pytorch
#          cu130 index (login internet). REQUIRED for shim_utils.h above.
#       2. install flash_attn_3 prebuilt wheel (cu130 torch2.11 aarch64).
#       3. install build-backend + runtime deps (offline wheelhouse; pure-python,
#          torch-agnostic -- they reuse the freshly-installed torch 2.11).
#       4. build vLLM editable --no-build-isolation --no-deps against torch 2.11,
#          with VLLM_CUTLASS_SRC_DIR -> staged v4.4.2 cutlass (no network for it).
#   + merge FlashQLA / tilelang / tvm_ffi / fla GDN overlay via debugfs rdump.
#   then apptainer build the final SIF.
#
#   *** TORCH-2.11 ABI CAVEAT ***  The base SIF's TE 2.7 / flash_attn 2.7.4 /
#   Megatron / apex were all COMPILED against torch 2.9. Bumping to 2.11 MAY break
#   their C-extension ABI. The SkyRL FSDP2 RL path this SIF is for does NOT use
#   Megatron, but TE / flash_attn could matter. The validation block at the end
#   imports torch/vllm/TE/flash_attn/megatron/apex + the FSDP2 train path and the
#   four model archs -- READ ITS OUTPUT to decide whether a TE/flash_attn rebuild
#   for torch 2.11 is needed before this SIF goes to production RL.
#
# OUTPUT: $CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
# =============================================================================
set -euo pipefail

CONTAINERS=/e/scratch/jureap59/feuer1/containers
BASE_SIF=$CONTAINERS/skyrl_megatron.sif
OVL_FLA=$CONTAINERS/fla_tilelang_overlay.img
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
VLLM_SRC=/e/scratch/jureap59/feuer1/sif_build/vllm_src          # fork @ v0.20.2rc0-305-g1948bebd1
CUTLASS_SRC=/e/scratch/jureap59/feuer1/vllm/.deps/cutlass-src   # pre-verified v4.4.2
BUILD_DEPS_WHEELS=/e/scratch/jureap59/feuer1/sif_build/build_deps_wheels
RUNTIME_DEPS_WHEELS=/e/scratch/jureap59/feuer1/sif_build/runtime_deps_wheels

STAMP=$(date +%Y%m%d_%H%M%S)
# Stage the ENTIRE build on the LOGIN-LOCAL /tmp (1.3TB, ~90M free inodes, NO
# project quota). The exa_scratch project quota is near its inode hard-limit
# (8.8M, ~1.67M free) -- a sandbox unpacks ~270k+ small files (cutlass docs etc.)
# and the first login attempt hit "Disk quota exceeded" mid-extraction on GPFS.
# Only the final OUT_SIF (a single ~10GB file) lands on the GPFS containers dir.
BUILD=/tmp/sifbuild0202_$STAMP
SANDBOX=$BUILD/sandbox
mkdir -p "$BUILD"
export APPTAINER_TMPDIR=$BUILD/aptmp
export APPTAINER_CACHEDIR=$BUILD/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
LOCAL=$BUILD/local
mkdir -p "$LOCAL"

echo "=== $(date) host=$(hostname) LOGIN-NODE build STAMP=$STAMP ==="
echo "BASE_SIF=$BASE_SIF ($(stat -c%s $BASE_SIF) bytes)"
echo "OVL_FLA=$OVL_FLA ($(stat -c%s $OVL_FLA) bytes)"
echo "VLLM_SRC=$VLLM_SRC"; cat "$VLLM_SRC/.vllm_commit" 2>/dev/null || true
echo "CUTLASS_SRC=$CUTLASS_SRC ($(cd $CUTLASS_SRC && git describe --tags 2>/dev/null))"
[ -d "$VLLM_SRC/vllm" ]      || { echo "FATAL: vLLM fork source not found at $VLLM_SRC"; exit 2; }
[ -d "$CUTLASS_SRC/include/cutlass" ] || { echo "FATAL: cutlass v4.4.2 src not found at $CUTLASS_SRC"; exit 2; }
ls "$BUILD_DEPS_WHEELS"/*.whl   >/dev/null 2>&1 || { echo "FATAL: no build-dep wheels in $BUILD_DEPS_WHEELS"; exit 4; }
ls "$RUNTIME_DEPS_WHEELS"/*.whl >/dev/null 2>&1 || { echo "FATAL: no runtime-dep wheels in $RUNTIME_DEPS_WHEELS"; exit 5; }
echo "BUILD (login-local /tmp)=$BUILD"; df -h /tmp | tail -1; df -i /tmp | tail -1

# Quick internet sanity (login node should reach github + pytorch).
echo "=== internet sanity ==="
timeout 10 curl -sI https://github.com 2>&1 | head -1 || echo "WARN: github unreachable"

# ---------------------------------------------------------------------------
# 1. Writable sandbox from base SIF
# ---------------------------------------------------------------------------
echo "=== [1] building sandbox from base SIF ==="
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "$BASE_SIF"
echo "sandbox built."

echo "=== [1b] staging vLLM fork source into sandbox ==="
mkdir -p "$SANDBOX/opt/vllm_build"
cp -a "$VLLM_SRC"/. "$SANDBOX/opt/vllm_build/"

echo "=== [1c] staging pre-verified cutlass v4.4.2 into sandbox ==="
mkdir -p "$SANDBOX/opt/cutlass_src"
cp -a "$CUTLASS_SRC"/. "$SANDBOX/opt/cutlass_src/"

echo "=== [1d] staging build-backend wheels into sandbox ==="
mkdir -p "$SANDBOX/opt/build_deps_wheels"
cp -a "$BUILD_DEPS_WHEELS"/. "$SANDBOX/opt/build_deps_wheels/"

echo "=== [1e] staging vLLM runtime-dep wheelhouse into sandbox ==="
mkdir -p "$SANDBOX/opt/runtime_deps_wheels"
cp -a "$RUNTIME_DEPS_WHEELS"/. "$SANDBOX/opt/runtime_deps_wheels/"
echo "staged $(ls "$SANDBOX/opt/runtime_deps_wheels"/*.whl | wc -l) runtime-dep wheels."

echo "=== [1f] pre-downloading flash_attn_3 prebuilt wheel (cu130 torch2.11 aarch64) on login node ==="
FA3_WHL=flash_attn_3-3.0.0+cu130torch2.11gite2743ab-cp39-abi3-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl
FA3_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.11/flash_attn_3-3.0.0%2Bcu130torch2.11gite2743ab-cp39-abi3-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl"
mkdir -p "$SANDBOX/opt/fa3_wheel"
wget -q "$FA3_URL" -O "$SANDBOX/opt/fa3_wheel/$FA3_WHL"
[ -s "$SANDBOX/opt/fa3_wheel/$FA3_WHL" ] || { echo "FATAL: flash_attn_3 wheel download failed"; exit 6; }
echo "flash_attn_3 wheel: $(stat -c%s "$SANDBOX/opt/fa3_wheel/$FA3_WHL") bytes"

# ---------------------------------------------------------------------------
# 2. Bump torch -> 2.11, install flash_attn_3, then build + install vLLM 0.20.2rc0
#    fork FROM SOURCE against torch 2.11. Runs INSIDE the sandbox ON THE LOGIN NODE
#    -> internet flows in for the torch 2.11 wheels. VLLM_CUTLASS_SRC_DIR -> staged
#    v4.4.2 cutlass src (no network for cutlass).
# ---------------------------------------------------------------------------
echo "=== [2] bump torch 2.11 + flash_attn_3 + build vLLM inside sandbox (login node) ==="
apptainer exec --writable --nv --no-home \
  --env MAX_JOBS=32 \
  --env NVCC_THREADS=8 \
  --env CMAKE_BUILD_TYPE=Release \
  --env VLLM_TARGET_DEVICE=cuda \
  --env TORCH_CUDA_ARCH_LIST="9.0+PTX" \
  --env VLLM_CUTLASS_SRC_DIR=/opt/cutlass_src \
  --env CCACHE_DISABLE=1 \
  --env PIP_NO_BUILD_ISOLATION=1 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc0 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.20.2rc0 \
  "$SANDBOX" bash -lc '
    set -euo pipefail
    cd /opt/vllm_build
    echo "--- pre-bump torch / nvcc ---"
    python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"abi\",torch._C._GLIBCXX_USE_CXX11_ABI)"
    nvcc --version | tail -2
    echo "--- BUMP torch -> 2.11.0+cu130 (REQUIRED for torch/headeronly/util/shim_utils.h) ---"
    # The fork csrc/cutlass_extensions/common.hpp needs the torch 2.11 headeronly
    # shim header absent from NGC torch 2.9 (Path A blocker). Install torch 2.11 +
    # matching torchvision/torchaudio from the pytorch cu130 index (login internet).
    python -m pip install "torch==2.11.0" torchvision torchaudio \
      --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --upgrade 2>&1 | tail -20 || \
    python -m pip install "torch==2.11.0" torchvision torchaudio \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --upgrade 2>&1 | tail -20
    python -c "import torch;print(\"POST-BUMP torch\",torch.__version__,\"cuda\",torch.version.cuda,\"abi\",torch._C._GLIBCXX_USE_CXX11_ABI)"
    echo "--- verify shim_utils.h is now present ---"
    ls "$(python -c "import torch,os;print(os.path.dirname(torch.__file__))")/include/torch/headeronly/util/shim_utils.h" \
      && echo "shim_utils.h PRESENT (torch 2.11)" || { echo "FATAL: shim_utils.h still missing after torch bump"; exit 7; }
    echo "--- install flash_attn_3 prebuilt wheel (--no-deps) ---"
    python -m pip install --no-deps /opt/fa3_wheel/*.whl
    echo "--- install PEP517 build-backend deps OFFLINE ---"
    python -m pip install --no-index --find-links /opt/build_deps_wheels \
      setuptools_scm setuptools wheel packaging
    python -c "import setuptools_scm; print(\"setuptools_scm import OK\")"
    python -c "import importlib.metadata as md; print(\"setuptools_scm\", md.version(\"setuptools-scm\"))" || true
    echo "--- strip ONLY the in-source torch pins via use_existing_torch (torch 2.11 already installed; this prevents a redundant 2.11 resolve from cuda.txt) ---"
    python use_existing_torch.py
    echo "--- install python build deps from requirements/build.txt (no torch) ---"
    [ -f requirements/build.txt ] && python -m pip install --no-build-isolation -r requirements/build.txt || echo "no requirements/build.txt (ok)"
    echo "--- install vLLM RUNTIME deps OFFLINE from the staged wheelhouse ---"
    # Pure-python / torch-agnostic; --find-links reuses the freshly-installed torch 2.11.
    python -m pip install --no-index --find-links /opt/runtime_deps_wheels \
      "cachetools" "blake3" "py-cpuinfo" "openai>=2.0.0" \
      "prometheus-fastapi-instrumentator>=7.0.0" "tiktoken>=0.6.0" \
      "lm-format-enforcer==0.11.3" "llguidance>=1.3.0,<1.4.0" "outlines_core==0.2.14" \
      "diskcache==5.6.3" "xgrammar>=0.2.0,<1.0.0" "partial-json-parser" "msgspec" \
      "gguf>=0.17.0" "mistral_common>=1.11.2" "compressed-tensors==0.15.0.1" \
      "depyf==0.20.0" "watchfiles" "pybase64" "cbor2" "ijson" "setproctitle" \
      "openai-harmony>=0.0.3" "anthropic>=0.71.0" \
      "model-hosting-container-standards>=0.1.14,<1.0.0" "mcp" \
      "opentelemetry-sdk>=1.27.0" "opentelemetry-exporter-otlp>=1.27.0" \
      "opentelemetry-semantic-conventions-ai>=0.4.1" "fastsafetensors>=0.2.2" || \
      echo "WARN: some runtime deps may need online resolve; vLLM editable --no-deps still proceeds"
    python -m pip install --no-deps --no-index --find-links /opt/runtime_deps_wheels \
      "opencv-python-headless>=4.13.0" || true
    echo "--- runtime-dep import smoke ---"
    python -c "import cachetools, blake3, openai, xgrammar, gguf, mistral_common, compressed_tensors, msgspec, mcp, fastsafetensors, cv2; print(\"runtime-dep imports OK\")" || echo "WARN: a runtime-dep import failed (see above)"
    echo "--- build + install vLLM (editable, no-build-isolation, --no-deps, torch 2.11, cutlass-from-src) ---"
    echo "    VLLM_CUTLASS_SRC_DIR=$VLLM_CUTLASS_SRC_DIR"
    python -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tail -250
    echo "--- post-install version probe ---"
    python -c "import vllm; print(\"vllm\", vllm.__version__)"
  '
echo "vLLM build/install complete."

# ---------------------------------------------------------------------------
# 3. Merge FlashQLA / tilelang / tvm_ffi / fla GDN overlay via debugfs rdump
# ---------------------------------------------------------------------------
merge_overlay() {
  local IMG=$1 NAME=$2
  echo "=== [3.$NAME] extracting $IMG via debugfs ==="
  local D=$LOCAL/x_$NAME
  rm -rf "$D"; mkdir -p "$D"
  debugfs -R "rdump /upper $D" "$IMG" 2>&1 | grep -v "^debugfs 1\." || true
  local UPPER="$D/upper"
  local n=$(find "$UPPER" -type f 2>/dev/null | wc -l)
  echo "$NAME extracted upper file count: $n"
  [ "$n" -gt 0 ] || { echo "FATAL: 0 files extracted for $NAME"; exit 3; }
  cp -a "$UPPER"/. "$SANDBOX"/
  echo "$NAME merged into sandbox."
  rm -rf "$D"
}
merge_overlay "$OVL_FLA" fla

echo "=== [3b] verifying merged GDN overlay paths ==="
for d in \
  usr/local/lib/python3.12/dist-packages/tilelang \
  usr/local/lib/python3.12/dist-packages/flash_qla \
  usr/local/lib/python3.12/dist-packages/tvm_ffi \
  usr/local/lib/python3.12/dist-packages/tilelang-0.1.8.dist-info \
  usr/local/lib/python3.12/dist-packages/flash_qla-0.1.0+6ef4858.dist-info \
  usr/local/lib/python3.12/dist-packages/apache_tvm_ffi-0.1.9.dist-info ; do
  [ -e "$SANDBOX/$d" ] && echo "OK  $d" || echo "MISSING $d"
done

echo "=== [3c] pruning vLLM build artifacts (keep source for editable import) ==="
rm -rf "$SANDBOX/opt/vllm_build/build" "$SANDBOX/opt/vllm_build/.deps" \
       "$SANDBOX/opt/cutlass_src" \
       "$SANDBOX"/root/.cache/pip "$SANDBOX"/opt/vllm_build/*.egg-info/SOURCES.txt 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Rebuild final SIF
# ---------------------------------------------------------------------------
echo "=== [4] building final SIF -> $OUT_SIF ==="
rm -f "$OUT_SIF"
apptainer build "$OUT_SIF" "$SANDBOX"
echo "final SIF built: $(stat -c%s $OUT_SIF) bytes"
ls -la "$OUT_SIF"

# ---------------------------------------------------------------------------
# 5. Validate
# ---------------------------------------------------------------------------
echo "=== [5] validating baked SIF ==="
apptainer exec --nv --env VLLM_ATTENTION_BACKEND=FLASH_ATTN "$OUT_SIF" python - <<'PYEOF' || echo "WARN: validation block exited non-zero (SIF already built; inspect output above)"
import importlib
def chk(n):
    try:
        m=importlib.import_module(n); print(f"IMPORT OK  {n}  v={getattr(m,'__version__','n/a')}"); return m
    except Exception as e:
        print(f"IMPORT FAIL {n}: {type(e).__name__}: {e}"); return None
print("--- base stack ---")
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
chk("vllm")
import vllm; print("VLLM_VERSION:", vllm.__version__)
chk("transformers")
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
print("transformers gemma4:", "gemma4" in C, "gemma4_text:", "gemma4_text" in C,
      "qwen3_next:", "qwen3_next" in C, "qwen3_moe:", "qwen3_moe" in C)
print("--- ABI-sensitive prebuilt stack (torch 2.11 bumped: were built vs 2.9 -- THIS IS THE DECISION POINT) ---")
chk("transformer_engine"); chk("flash_attn")
try:
    import megatron.core as mc; print("IMPORT OK  megatron.core v=", getattr(mc,"__version__","n/a"))
except Exception as e: print("megatron.core:", type(e).__name__, e)
try:
    import apex; print("IMPORT OK  apex")
except Exception as e: print("apex:", type(e).__name__, e)
chk("skyrl_train"); chk("skyrl_gym")
print("--- GDN overlay ---")
chk("tilelang"); chk("tvm_ffi")
try:
    import flash_qla; print("IMPORT OK flash_qla", getattr(flash_qla,'__version__','n/a'))
except Exception as e: print("flash_qla:", type(e).__name__, e)
print("--- vLLM model registry archs ---")
try:
    from vllm import ModelRegistry
    archs=set(ModelRegistry.get_supported_archs())
    for a in ("Gemma4ForConditionalGeneration","Gemma4ForCausalLM","Qwen3MoeForCausalLM","Qwen3NextForCausalLM"):
        print(f"  REGISTRY {a}: {a in archs}")
except Exception as e: print("registry probe FAIL:", type(e).__name__, e)
print("--- R3 routed_experts native capture (serving + protocol) ---")
import os
sp="/opt/vllm_build"
for rel in ("vllm/entrypoints/openai/chat_completion/serving.py",
            "vllm/entrypoints/openai/chat_completion/protocol.py",
            "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"):
    p=os.path.join(sp,rel)
    if os.path.exists(p):
        print(f"  {rel}: routed_experts_count={open(p).read().count('routed_experts')}")
    else:
        import vllm as _v
        ip=os.path.join(os.path.dirname(_v.__file__), rel.split('vllm/',1)[1])
        print(f"  {rel}: src-missing installed_exists={os.path.exists(ip)} "
              f"count={open(ip).read().count('routed_experts') if os.path.exists(ip) else 'n/a'}")
print("--- GDN overlay version parity ---")
import importlib.metadata as md
def ver(p):
    try: return md.version(p)
    except Exception as e: return f"ERR:{e}"
for pkg,exp in [("tilelang","0.1.8"),("apache-tvm-ffi","0.1.9"),("flash_qla","0.1.0+6ef4858")]:
    print(f"  {pkg} = {ver(pkg)} (expect {exp})")
print("--- runtime-dep closure smoke ---")
for n in ("cachetools","blake3","openai","xgrammar","gguf","mistral_common",
          "compressed_tensors","msgspec","mcp","fastsafetensors","cbor2","ijson",
          "pybase64","setproctitle","depyf","watchfiles","tiktoken","diskcache",
          "lm_format_enforcer","llguidance","outlines_core","partial_json_parser",
          "openai_harmony","anthropic","prometheus_fastapi_instrumentator",
          "opentelemetry.sdk","opentelemetry.exporter.otlp","cv2","py_cpuinfo"):
    chk(n)
try:
    from vllm.entrypoints.openai.api_server import build_app
    print("IMPORT OK  vllm.entrypoints.openai.api_server.build_app")
except Exception as e:
    print(f"IMPORT FAIL vllm.entrypoints.openai.api_server: {type(e).__name__}: {e}")
try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    print("IMPORT OK  vllm.engine.arg_utils.AsyncEngineArgs")
except Exception as e:
    print(f"IMPORT FAIL vllm.engine.arg_utils: {type(e).__name__}: {e}")
print("--- tiny-model load smoke (FLASH_ATTN backend; needs GPU + cached model) ---")
import glob as _g
cand=[]
for root in (os.environ.get("HF_HUB_CACHE",""), "/e/data1/datasets/playground/ot-baf/hf_hub",
             os.path.expanduser("~/.cache/huggingface/hub")):
    if root and os.path.isdir(root):
        for pat in ("Qwen3-0.6B","Qwen2.5-0.5B","Qwen3-0.5B","0.5B","0.6B"):
            cand += _g.glob(os.path.join(root,"models--*"+pat.replace('.','')+"*"))
            cand += _g.glob(os.path.join(root,"models--*"+pat+"*"))
cand=[c for c in cand if os.path.isdir(c)]
if not torch.cuda.is_available():
    print("  SKIP tiny-model smoke: no GPU visible on build node")
elif not cand:
    print("  SKIP tiny-model smoke: no tiny model in HF cache (will smoke at serve time)")
else:
    try:
        snaps=_g.glob(os.path.join(cand[0],"snapshots","*"))
        mp=snaps[0] if snaps else cand[0]
        from vllm import LLM, SamplingParams
        llm=LLM(model=mp, max_model_len=512, gpu_memory_utilization=0.3, enforce_eager=True)
        out=llm.generate(["Hello"], SamplingParams(max_tokens=4))
        print(f"  TINY-MODEL LOAD OK ({os.path.basename(cand[0])}): generated {len(out)} seq")
    except Exception as e:
        print(f"  TINY-MODEL LOAD FAIL: {type(e).__name__}: {e}")
PYEOF

echo "=== $(date) BUILD COMPLETE ==="
echo "OUT_SIF=$OUT_SIF"
echo "=== removing login-local build scratch ($BUILD) ==="
rm -rf "$BUILD" 2>/dev/null || true
echo "=== done ==="
