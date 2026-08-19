#!/bin/bash
# =============================================================================
# build_vllm0202rc0_r3_sif.sbatch
#
# Reproduces the (purged) #208/#211 "shelf asset": a SkyRL+Megatron+TE SIF whose
# vLLM is the penfever fork at v0.20.2rc0 + native R3 routed-experts capture, with
# Gemma4 / Qwen3-MoE / Qwen3-Next model support, on Jupiter GH200 (aarch64, cu13).
#
# STRATEGY
#   base SIF  = skyrl_megatron.sif      (NGC 25.09: torch 2.9.0a0+nv25.09 cu130,
#                                        triton 3.4.0, Megatron-core 0.14.0, TE 2.7,
#                                        apex, flash_attn 2.7.4, transformers 5.10.1
#                                        [gemma4/qwen3_next/qwen3_moe], SkyRL editable
#                                        at /opt/SkyRL, nvcc CUDA 13.0). NO vLLM.
#   + build the vLLM 0.20.2rc0 fork FROM SOURCE against the in-SIF torch 2.9
#     (use_existing_torch.py strips the fork's torch==2.11.0 pin so we keep the
#     NGC torch the whole Megatron/TE/apex/flash_attn/flashinfer stack is built on).
#     The fork's HEAD (v0.20.2rc0-305-g1948bebd1) already carries the R3 capture
#     rail (routed_experts_capturer.py + routed_experts emission in the four
#     chat/completion serving+protocol files) and the gemma4 models — so NO extra
#     R3 patch is applied; R3 is native.
#   + merge the FlashQLA / tilelang / tvm_ffi / fla GDN overlay (fla_tilelang_overlay.img)
#     via debugfs rdump (no fuse, no root) — same as bake_r3_sif.sbatch.
#   then apptainer build the final SIF.
#
# OUTPUT: $CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
#
# The fork SOURCE is rsync'd to the cluster separately (per CLAUDE.md, vLLM syncs
# via rsync, not git-to-cluster). This sbatch expects it at $VLLM_SRC below.
# =============================================================================
#SBATCH --job-name=build_vllm0202rc0_r3
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=/e/scratch/jureap59/feuer1/sif_build/logs/build0202_%j.out
#SBATCH --error=/e/scratch/jureap59/feuer1/sif_build/logs/build0202_%j.err

set -euo pipefail

CONTAINERS=/e/scratch/jureap59/feuer1/containers
BASE_SIF=$CONTAINERS/skyrl_megatron.sif                 # NGC base, NO vLLM
OVL_FLA=$CONTAINERS/fla_tilelang_overlay.img            # FlashQLA/tilelang/tvm_ffi/fla GDN
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
VLLM_SRC=/e/scratch/jureap59/feuer1/sif_build/vllm_src  # rsync'd fork @ v0.20.2rc0-305-g1948bebd1

# Build scratch on GPFS (node-local /tmp too small for sandbox+extract+SIF).
BUILD=/e/scratch/jureap59/feuer1/sif_build/job_$SLURM_JOB_ID
SANDBOX=$BUILD/sandbox
mkdir -p "$BUILD"
export APPTAINER_TMPDIR=$BUILD/aptmp
export APPTAINER_CACHEDIR=$BUILD/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
LOCAL=/tmp/sif_build_$SLURM_JOB_ID
mkdir -p "$LOCAL"

cleanup() { rm -rf "$LOCAL" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
echo "BASE_SIF=$BASE_SIF ($(stat -c%s $BASE_SIF) bytes)"
echo "OVL_FLA=$OVL_FLA ($(stat -c%s $OVL_FLA) bytes)"
echo "VLLM_SRC=$VLLM_SRC"
[ -d "$VLLM_SRC/vllm" ] || { echo "FATAL: vLLM fork source not found at $VLLM_SRC (rsync it first)"; exit 2; }
cat "$VLLM_SRC/.vllm_commit" 2>/dev/null || true
df -h /tmp | tail -1; df -h /e/scratch | tail -1

# ---------------------------------------------------------------------------
# 1. Writable sandbox from base SIF
# ---------------------------------------------------------------------------
echo "=== [1] building sandbox from base SIF ==="
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "$BASE_SIF"
echo "sandbox built."

# Stage the fork source INTO the sandbox (writable; pip build writes egg-info etc.)
echo "=== [1b] staging vLLM fork source into sandbox ==="
mkdir -p "$SANDBOX/opt/vllm_build"
cp -a "$VLLM_SRC"/. "$SANDBOX/opt/vllm_build/"

# Stage the PEP517 build-backend wheels (setuptools_scm + deps) INTO the sandbox.
# The booster compute node has NO internet, so pip cannot fetch these at build
# time. They are pre-downloaded on the (internet-enabled) login node into
# $BUILD_DEPS_WHEELS and installed offline (--no-index) before the editable
# vLLM build, which runs setup.py in-env under --no-build-isolation and needs
# setuptools_scm present. (Root cause of the job 848577 failure.)
BUILD_DEPS_WHEELS=/e/scratch/jureap59/feuer1/sif_build/build_deps_wheels
echo "=== [1c] staging build-backend wheels into sandbox ==="
[ -d "$BUILD_DEPS_WHEELS" ] || { echo "FATAL: build-deps wheelhouse not found at $BUILD_DEPS_WHEELS (pre-download on login node: apptainer exec $BASE_SIF python -m pip download --dest $BUILD_DEPS_WHEELS setuptools_scm setuptools wheel packaging)"; exit 4; }
ls "$BUILD_DEPS_WHEELS"/*.whl >/dev/null 2>&1 || { echo "FATAL: no .whl in $BUILD_DEPS_WHEELS"; exit 4; }
mkdir -p "$SANDBOX/opt/build_deps_wheels"
cp -a "$BUILD_DEPS_WHEELS"/. "$SANDBOX/opt/build_deps_wheels/"
echo "staged $(ls "$SANDBOX/opt/build_deps_wheels"/*.whl | wc -l) build-dep wheels."

# Stage the vLLM RUNTIME-dependency wheelhouse INTO the sandbox. The editable
# vLLM install runs with --no-deps (see step 2 below) precisely so pip never
# resolves cuda.txt/common.txt against PyPI -- the cuda.txt torch==2.11 /
# tilelang==0.1.9 pins would otherwise clobber the NGC torch 2.9 + overlay
# tilelang 0.1.8. But that means vLLM's *pure-python* runtime deps (cachetools,
# blake3, openai, xgrammar, mcp, opentelemetry, ...) are NOT installed by the
# editable build. The base SIF carries only 27/59 of common.txt; 32 are missing.
# They are pre-downloaded on the (internet-enabled) login node into
# $RUNTIME_DEPS_WHEELS (using the base SIF's pip for ABI/version match) and
# installed offline (--no-index) BEFORE the editable build. The wheelhouse
# deliberately EXCLUDES torch/triton/transformers/tokenizers/numpy/pillow/nvidia
# wheels (all already in the base SIF at the correct NGC versions) so the offline
# resolver reuses the in-SIF stack and never bumps numpy to 2.x (which would
# break the Megatron/TE/apex build). (Root cause of the job 849897 failure:
# `pip install -e .` reached out to PyPI for cachetools and hit "Network is
# unreachable" on the offline node.)
RUNTIME_DEPS_WHEELS=/e/scratch/jureap59/feuer1/sif_build/runtime_deps_wheels
echo "=== [1d] staging vLLM runtime-dep wheelhouse into sandbox ==="
[ -d "$RUNTIME_DEPS_WHEELS" ] || { echo "FATAL: runtime-deps wheelhouse not found at $RUNTIME_DEPS_WHEELS (pre-download on login node: apptainer exec $BASE_SIF python -m pip download --dest $RUNTIME_DEPS_WHEELS <the 32 missing common.txt pure-python deps>, then prune torch/triton/transformers/tokenizers/numpy/pillow/nvidia_*/cuda_* wheels)"; exit 5; }
ls "$RUNTIME_DEPS_WHEELS"/*.whl >/dev/null 2>&1 || { echo "FATAL: no .whl in $RUNTIME_DEPS_WHEELS"; exit 5; }
mkdir -p "$SANDBOX/opt/runtime_deps_wheels"
cp -a "$RUNTIME_DEPS_WHEELS"/. "$SANDBOX/opt/runtime_deps_wheels/"
echo "staged $(ls "$SANDBOX/opt/runtime_deps_wheels"/*.whl | wc -l) runtime-dep wheels."

# ---------------------------------------------------------------------------
# 2. Build + install vLLM 0.20.2rc0 fork FROM SOURCE against in-SIF torch 2.9
# ---------------------------------------------------------------------------
echo "=== [2] building vLLM from source inside sandbox (against existing torch) ==="
# --writable-tmpfs is not enough (pip needs to write into /usr/local/lib); use
# --writable on the sandbox. Bind nothing external; compile fully offline.
# MAX_JOBS bounded to keep peak RSS under the node; nvcc + ninja from the SIF.
# NOTE: the rsync'd source has NO .git, so setuptools_scm cannot derive the
# version. Pin it explicitly so the wheel reports 0.20.2rc0 (matches the fork's
# `git describe` tag v0.20.2rc0). Without this, the build errors or reports 0.0.0.
apptainer exec --writable --no-home \
  --env MAX_JOBS=48 \
  --env CMAKE_BUILD_TYPE=Release \
  --env VLLM_TARGET_DEVICE=cuda \
  --env TORCH_CUDA_ARCH_LIST="9.0+PTX" \
  --env CCACHE_DISABLE=1 \
  --env PIP_NO_BUILD_ISOLATION=1 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc0 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.20.2rc0 \
  "$SANDBOX" bash -lc '
    set -euo pipefail
    cd /opt/vllm_build
    echo "--- pre-build torch / nvcc ---"
    python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda)"
    nvcc --version | tail -2
    echo "--- strip torch pins so the build uses the in-SIF NGC torch 2.9 ---"
    python use_existing_torch.py
    echo "--- install PEP517 build-backend deps OFFLINE (setuptools_scm + deps) ---"
    # With --no-build-isolation the editable build runs setup.py in THIS env, so
    # setuptools_scm (+ setuptools/wheel/packaging) must be present here explicitly.
    # The base SIF lacks setuptools_scm -> job 848577 failed at "Preparing editable
    # metadata (pyproject.toml)" with ModuleNotFoundError: No module named
    # setuptools_scm. The compute node has no internet; install from the staged
    # wheelhouse. NOT guarded by `|| true` -- this MUST succeed or the build is
    # doomed at metadata generation anyway.
    python -m pip install --no-index --find-links /opt/build_deps_wheels \
      setuptools_scm setuptools wheel packaging
    # VERIFY: just confirm the import succeeds + report the version via the
    # packaging-metadata API. setuptools_scm v8 REMOVED the module-level
    # `__version__` attribute, so `setuptools_scm.__version__` AttributeErrors
    # under set -e (job 849267). Use importlib.metadata, and do NOT gate the
    # build on the version string -- a successful import is the only thing the
    # editable build actually needs.
    # NB: the whole in-sandbox block is wrapped in bash -lc + a single-quoted string,
    # so these python -c snippets must use ESCAPED DOUBLE QUOTES (\") ONLY. Any bare
    # single-quote char anywhere in this block (even in a comment) closes the outer
    # quote and the shell runs import as a command (job 849776: unexpected EOF).
    python -c "import setuptools_scm; print(\"setuptools_scm import OK\")"  # the REAL gate (import must work)
    python -c "import importlib.metadata as md; print(\"setuptools_scm\", md.version(\"setuptools-scm\"))" || true  # version report is non-fatal
    echo "--- install python build deps (no torch) ---"
    python -m pip install --no-build-isolation -r requirements/build.txt || true
    echo "--- install vLLM RUNTIME deps OFFLINE from the staged wheelhouse ---"
    # Two passes, both --no-index (offline). The wheelhouse has the 32 missing
    # common.txt pure-python deps + their pure-python transitives, and EXCLUDES
    # torch/triton/transformers/tokenizers/numpy/pillow/nvidia (already in-SIF).
    # PASS A: the 30 named deps the resolver can satisfy against the in-SIF stack
    #   (mistral_common installed WITHOUT its [image] extra -- that extra pulls
    #   opencv-python which pins numpy>=2 and is unsatisfiable against the in-SIF
    #   numpy 1.26.4). Resolver ON so transitive pure-python deps (jiter, distro,
    #   httpx, starlette, sse-starlette, opentelemetry-proto, ...) come along.
    #   NOT guarded by `|| true` -- a missing runtime import dooms vLLM at serve
    #   time, so this MUST succeed.
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
      "opentelemetry-semantic-conventions-ai>=0.4.1" "fastsafetensors>=0.2.2"
    # PASS B: opencv-python-headless --no-deps. Its >=2 numpy pin is conservative;
    # it runs fine on the in-SIF numpy 1.26.4. --no-deps so it does NOT try to
    # upgrade numpy (which would break the Megatron/TE/apex stack).
    python -m pip install --no-deps --no-index --find-links /opt/runtime_deps_wheels \
      "opencv-python-headless>=4.13.0"
    echo "--- runtime-dep import smoke (offline) ---"
    python -c "import cachetools, blake3, openai, xgrammar, gguf, mistral_common, compressed_tensors, msgspec, mcp, fastsafetensors, cv2; print(\"runtime-dep imports OK\")"
    echo "--- build + install vLLM (editable, no-build-isolation, --no-deps, existing torch) ---"
    # precompiled kernels are unavailable for this torch; compile from source.
    # --no-deps is CRITICAL: without it pip resolves requirements/cuda.txt against
    # PyPI, whose torch==2.11.0 / tilelang==0.1.9 pins would clobber the NGC torch
    # 2.9 + overlay tilelang 0.1.8 (and the offline node cannot reach PyPI anyway
    # -> job 849897 "Network is unreachable" / "No matching distribution for
    # cachetools"). The runtime deps were just installed offline above, so vLLM
    # has everything it needs WITHOUT letting pip touch the dependency graph.
    python -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tail -120
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

# Drop the bulky source tree from the final image (keep the editable install
# pointer though — editable -e installs reference /opt/vllm_build/vllm at import
# time, so we KEEP the source but prune build artifacts to shrink the SIF).
echo "=== [3c] pruning vLLM build artifacts (keep source for editable import) ==="
rm -rf "$SANDBOX/opt/vllm_build/build" "$SANDBOX/opt/vllm_build/.deps" \
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
# Post-build validation is informational only -- the SIF is already written at
# this point, so a probe hiccup (e.g. no GPU visible on the build node for
# torch.cuda.is_available, or an unexpected import surprise) must NOT flip the
# whole job to FAILED and mask a successful build. Guarded with `|| true`.
#
# FLASHINFER DEFERRED: this SIF is built WITHOUT flashinfer-python (its dep tree
# -- nvidia-cutlass-dsl/cuda-tile/nvidia-cudnn-frontend -- is heavy and would
# drag numpy>=2, breaking the in-SIF numpy 1.26 stack; flashinfer also JIT-builds
# cubins at runtime which fails on the offline compute node). The base SIF has
# flash_attn 2.7.4.post1, so we force VLLM_ATTENTION_BACKEND=FLASH_ATTN for both
# the smoke below and all production runs with this SIF. flashinfer is an
# OPTIONAL follow-up (would need a prebuilt cu13 wheel + its dep closure).
apptainer exec --nv --env VLLM_ATTENTION_BACKEND=FLASH_ATTN "$OUT_SIF" python - <<'PYEOF' || echo "WARN: validation block exited non-zero (SIF already built; inspect output above)"
import importlib
def chk(n):
    try:
        m=importlib.import_module(n); print(f"IMPORT OK  {n}  v={getattr(m,'__version__','n/a')}"); return m
    except Exception as e:
        print(f"IMPORT FAIL {n}: {type(e).__name__}: {e}"); return None
print("--- base stack ---")
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
v=chk("vllm")
import vllm; print("VLLM_VERSION:", vllm.__version__)
chk("transformers")
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
print("transformers gemma4:", "gemma4" in C, "gemma4_text:", "gemma4_text" in C,
      "qwen3_next:", "qwen3_next" in C, "qwen3_moe:", "qwen3_moe" in C)
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
sp="/opt/vllm_build"  # editable source root
for rel in ("vllm/entrypoints/openai/chat_completion/serving.py",
            "vllm/entrypoints/openai/chat_completion/protocol.py",
            "vllm/entrypoints/openai/completion/serving.py",
            "vllm/entrypoints/openai/completion/protocol.py",
            "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"):
    p=os.path.join(sp,rel)
    if os.path.exists(p):
        print(f"  {rel}: routed_experts_count={open(p).read().count('routed_experts')}")
    else:
        # fall back to installed location
        import vllm as _v, pathlib
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
print("--- runtime-dep closure smoke (catches --no-deps gaps) ---")
# --no-deps on the editable vLLM install means pip never validated vLLM's
# runtime import graph. Exercise the heaviest import paths (the OpenAI serving
# entrypoint + engine args + the runtime deps) so any STILL-missing module
# surfaces here as an actionable name, not at first serve.
for n in ("cachetools","blake3","openai","xgrammar","gguf","mistral_common",
          "compressed_tensors","msgspec","mcp","fastsafetensors","cbor2","ijson",
          "pybase64","setproctitle","depyf","watchfiles","tiktoken","diskcache",
          "lm_format_enforcer","llguidance","outlines_core","partial_json_parser",
          "openai_harmony","anthropic","prometheus_fastapi_instrumentator",
          "opentelemetry.sdk","opentelemetry.exporter.otlp","cv2","py_cpuinfo"):
    chk(n)
try:
    from vllm.entrypoints.openai.api_server import build_app  # heavy serving import chain
    print("IMPORT OK  vllm.entrypoints.openai.api_server.build_app")
except Exception as e:
    print(f"IMPORT FAIL vllm.entrypoints.openai.api_server: {type(e).__name__}: {e}")
try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    print("IMPORT OK  vllm.engine.arg_utils.AsyncEngineArgs")
except Exception as e:
    print(f"IMPORT FAIL vllm.engine.arg_utils: {type(e).__name__}: {e}")
print("--- tiny-model load smoke (FLASH_ATTN backend; needs a GPU + a cached model) ---")
# Confirms the --no-deps install can actually stand up an engine. Looks for any
# tiny model already in the HF cache (offline build node has no internet); if
# none is found the smoke is SKIPPED (not failed) -- the import closure above is
# the load-bearing --no-deps gap check, and a real engine smoke runs at serve
# time on the cluster anyway.
import glob as _g
cand=[]
for root in (os.environ.get("HF_HUB_CACHE",""), "/e/data1/datasets/playground/ot-baf/hf_hub",
             os.path.expanduser("~/.cache/huggingface/hub")):
    if root and os.path.isdir(root):
        for pat in ("*Qwen3-0.6B*","*Qwen2.5-0.5B*","*Qwen3-0.5B*","*tiny*","*0.5B*","*0.6B*"):
            cand += _g.glob(os.path.join(root,"models--*"+"".join(c for c in pat if c not in "*")+"*"))
cand=[c for c in cand if os.path.isdir(c)]
if not torch.cuda.is_available():
    print("  SKIP tiny-model smoke: no GPU visible on build node")
elif not cand:
    print("  SKIP tiny-model smoke: no tiny model in HF cache (offline; will smoke at serve time)")
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
echo "=== removing sandbox ==="
rm -rf "$SANDBOX" "$BUILD/aptmp" "$BUILD/apcache" 2>/dev/null || true
echo "=== done ==="
