#!/usr/bin/env bash
set -euo pipefail

# Disposable arm64 kernel gate. This tests TE/FA4 directly, not the complete
# project lock or a Grug worker.
wheel_uri="${1:?usage: preflight_gb200.sh S3_WHEEL_URI SHA256}"
wheel_sha256="${2:?usage: preflight_gb200.sh S3_WHEEL_URI SHA256}"
wheel=/tmp/transformer_engine_torch-2.19.0-cp312-cp312-linux_aarch64.whl
venv=/tmp/fa4-gb200-preflight

uv run --no-project --with boto3==1.42.97 python experiments/fa4/download_candidate.py \
    "$wheel_uri" "$wheel" "$wheel_sha256"
uv venv --python 3.12.14 "$venv"
# Run this disposable resolver outside the project so the kernel preflight is
# independent of the evolving arm64 project closure.
(
    cd /tmp
    uv pip install --python "$venv/bin/python" --only-binary all \
        --extra-index-url https://download.pytorch.org/whl/cu132 \
        --index-strategy unsafe-best-match \
        'torch==2.13.0+cu132' \
        'transformer-engine[pytorch]==2.19.0' \
        'transformer-engine-cu13==2.19.0' \
        "$wheel" \
        'flash-attn-4[cu13]==4.0.0b31' \
        'nvidia-cutlass-dsl==4.6.2' \
        'quack-kernels==0.6.4' \
        'apache-tvm-ffi==0.1.14.post0'
)

"$venv/bin/python" -c 'import torch, transformer_engine.pytorch, flash_attn.cute.interface; print(torch.__version__, torch.cuda.get_device_name())'
env NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 \
    NVTE_FLASH_ATTN_V4=1 timeout 600 "$venv/bin/python" experiments/fa4/probe_attention.py \
    --output /tmp/fa4-gb200-attention.pt --batch 1 --seq 32 --heads 2 --kv-heads 1 \
    --head-dim 64 --window-left 16 --warmups 1 --samples 2
