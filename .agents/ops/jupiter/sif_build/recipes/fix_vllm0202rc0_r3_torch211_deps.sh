#!/bin/bash
# =============================================================================
# fix_vllm0202rc0_r3_torch211_deps.sh   (LOGIN-NODE corrective rebuild)
#
# The torch-2.11 vLLM build (build_vllm0202rc0_r3_login_torch211.sh) succeeded:
# vllm 0.20.2rc0 compiled + installed, R3 native, GDN overlay, 4 archs registered.
# But the torch-2.11 bump left 3 gaps the validation surfaced:
#   1. flash_attn 2.7.4 (NGC, built vs torch 2.9) -> undefined-symbol ABI break.
#      SkyRL FSDP2 model_wrapper.py:20 does an UNCONDITIONAL
#      `from flash_attn.bert_padding import pad_input, unpad_input` -> the entire
#      FSDP2 RL worker fails to import. THIS IS RL-BLOCKING. Fix: install the
#      torch-2.11-matched prebuilt FA2 2.6.3 (cu130torch2.11 cp312 aarch64) from
#      mjun0812 v0.9.22. 2.6.3 <= 2.7.4.post1 satisfies SkyRL's version guard and
#      exposes the same bert_padding pad_input/unpad_input API.
#   2. uvloop missing -> vllm.entrypoints.openai.api_server import fails (serving).
#   3. hydra-core missing -> skyrl_train.entrypoints.main_base import fails.
# (transformer_engine 2.7 / megatron.core ALSO break under torch 2.11, but the
#  SkyRL FSDP2 path does NOT import them -- verified -- so that break is tolerable;
#  only the unused Megatron-backend RL would care. Left as-is.)
#
# This recreates a sandbox from the ALREADY-BUILT SIF (vLLM stays compiled, NO
# recompile) on login-local /tmp, installs the 3 deps (login internet), then
# rebuilds the SIF in place.
# =============================================================================
set -euo pipefail

CONTAINERS=/e/scratch/jureap59/feuer1/containers
IN_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif   # rebuild in place

STAMP=$(date +%Y%m%d_%H%M%S)
BUILD=/tmp/siffix0202_$STAMP
SANDBOX=$BUILD/sandbox
mkdir -p "$BUILD"
export APPTAINER_TMPDIR=$BUILD/aptmp
export APPTAINER_CACHEDIR=$BUILD/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "=== $(date) host=$(hostname) CORRECTIVE dep-fix STAMP=$STAMP ==="
echo "IN_SIF=$IN_SIF ($(stat -c%s $IN_SIF) bytes)"
df -h /tmp | tail -1

echo "=== [1] sandbox from already-built SIF (vLLM stays compiled) ==="
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "$IN_SIF"

echo "=== [1b] pre-download torch-2.11-matched FA2 2.6.3 wheel on login node ==="
FA2_WHL=flash_attn-2.6.3+cu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl
FA2_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/flash_attn-2.6.3%2Bcu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl"
mkdir -p "$SANDBOX/opt/fa2_wheel"
wget -q "$FA2_URL" -O "$SANDBOX/opt/fa2_wheel/$FA2_WHL"
[ -s "$SANDBOX/opt/fa2_wheel/$FA2_WHL" ] || { echo "FATAL: FA2 2.6.3 wheel download failed"; exit 6; }
echo "FA2 wheel: $(stat -c%s "$SANDBOX/opt/fa2_wheel/$FA2_WHL") bytes"

echo "=== [2] installing FA2 2.6.3 + uvloop + hydra-core inside sandbox (login node) ==="
apptainer exec --writable --nv --no-home "$SANDBOX" bash -lc '
  set -euo pipefail
  echo "--- pre torch ---"; python -c "import torch;print(torch.__version__)"
  echo "--- replace broken FA2 with torch-2.11 build (--no-deps; keep torch 2.11) ---"
  python -m pip install --no-deps --force-reinstall /opt/fa2_wheel/*.whl
  echo "--- install uvloop + hydra-core (online; pure runtime deps) ---"
  python -m pip install "uvloop" "hydra-core" "omegaconf"
  echo "--- import smokes ---"
  python -c "from flash_attn.bert_padding import pad_input, unpad_input; import flash_attn; print(\"flash_attn\", flash_attn.__version__, \"bert_padding OK\")"
  python -c "import uvloop; print(\"uvloop OK\")"
  python -c "import hydra; print(\"hydra OK\")"
  python -c "import skyrl_train.workers.fsdp.fsdp_worker; print(\"FSDP2 worker import OK\")"
  python -c "from vllm.entrypoints.openai.api_server import build_app; print(\"vllm api_server build_app OK\")"
  python -c "import vllm; print(\"vllm\", vllm.__version__)"
'

echo "=== [2b] prune wheel staging ==="
rm -rf "$SANDBOX/opt/fa2_wheel" "$SANDBOX"/root/.cache/pip 2>/dev/null || true

echo "=== [3] rebuild SIF in place -> $OUT_SIF ==="
TMP_OUT=$BUILD/out.sif
apptainer build "$TMP_OUT" "$SANDBOX"
mv -f "$TMP_OUT" "$OUT_SIF"
echo "SIF rebuilt: $(stat -c%s $OUT_SIF) bytes"; ls -la "$OUT_SIF"

echo "=== [4] final validation ==="
apptainer exec --nv --env VLLM_ATTENTION_BACKEND=FLASH_ATTN "$OUT_SIF" python - <<'PYEOF' || echo "WARN: validation exited non-zero (SIF already written)"
import importlib
def chk(n):
    try:
        m=importlib.import_module(n); print(f"IMPORT OK  {n}  v={getattr(m,'__version__','n/a')}")
    except Exception as e:
        print(f"IMPORT FAIL {n}: {type(e).__name__}: {str(e)[:90]}")
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)
import vllm; print("vllm", vllm.__version__)
chk("flash_attn"); chk("flash_attn_3"); chk("uvloop"); chk("hydra")
chk("transformer_engine")
try:
    import skyrl_train.workers.fsdp.fsdp_worker; print("IMPORT OK  skyrl_train.workers.fsdp.fsdp_worker (FSDP2 RL path)")
except Exception as e: print("FAIL FSDP2 worker:", type(e).__name__, str(e)[:90])
try:
    from vllm.entrypoints.openai.api_server import build_app; print("IMPORT OK  vllm api_server build_app")
except Exception as e: print("FAIL api_server:", type(e).__name__, str(e)[:90])
from vllm import ModelRegistry
archs=set(ModelRegistry.get_supported_archs())
for a in ("Gemma4ForConditionalGeneration","Gemma4ForCausalLM","Qwen3MoeForCausalLM","Qwen3NextForCausalLM"):
    print(f"  REGISTRY {a}: {a in archs}")
PYEOF

echo "=== removing build scratch ==="
rm -rf "$BUILD" 2>/dev/null || true
echo "=== $(date) DEP-FIX COMPLETE  OUT_SIF=$OUT_SIF ==="
