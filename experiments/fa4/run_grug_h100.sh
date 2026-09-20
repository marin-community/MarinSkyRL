#!/usr/bin/env bash
set -euo pipefail

world_size="${1:?usage: run_grug_h100.sh WORLD_SIZE SHAPE [STEPS]}"
shape="${2:?usage: run_grug_h100.sh WORLD_SIZE SHAPE [STEPS]}"
steps="${3:-1}"
uv sync --frozen --extra vllm --extra megatron --extra fa4 --group dev

common_env=(PYTHONPATH=skyrl-train NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 RAY_DEDUP_LOGS=0)
common_args=(--world-size "$world_size" --shape "$shape" --steps "$steps")
env "${common_env[@]}" NVTE_FLASH_ATTN_V4=0 timeout 900 .venv/bin/python \
    experiments/fa4/grug_step.py --output /tmp/fa4-grug-fa2.pt "${common_args[@]}"
env "${common_env[@]}" NVTE_FLASH_ATTN_V4=1 timeout 900 .venv/bin/python \
    experiments/fa4/grug_step.py --output /tmp/fa4-grug-fa4.pt "${common_args[@]}"
.venv/bin/python experiments/fa4/compare_grug.py /tmp/fa4-grug-fa2.pt /tmp/fa4-grug-fa4.pt
