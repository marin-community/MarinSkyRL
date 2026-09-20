#!/usr/bin/env bash
set -euo pipefail

# Continue past a failed HF parity assertion only to preserve FA2/FA4 data.
# The task exits nonzero if either arm fails, even when comparison succeeds.
world_size="${1:?usage: run_diagnostic_pair.sh WORLD_SIZE SHAPE STEPS}"
shape="${2:?usage: run_diagnostic_pair.sh WORLD_SIZE SHAPE STEPS}"
steps="${3:?usage: run_diagnostic_pair.sh WORLD_SIZE SHAPE STEPS}"
uv sync --frozen --extra vllm --extra megatron --extra fa4 --group dev

common_env=(PYTHONPATH=skyrl-train NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 RAY_DEDUP_LOGS=0)
common_args=(--world-size "$world_size" --shape "$shape" --steps "$steps" --diagnose-parity-failure)
fa2_status=0
env "${common_env[@]}" NVTE_FLASH_ATTN_V4=0 timeout 900 .venv/bin/python \
    experiments/fa4/grug_step.py --output /tmp/fa4-grug-fa2.pt "${common_args[@]}" || fa2_status=$?
test -s /tmp/fa4-grug-fa2.pt
fa4_status=0
env "${common_env[@]}" NVTE_FLASH_ATTN_V4=1 timeout 900 .venv/bin/python \
    experiments/fa4/grug_step.py --output /tmp/fa4-grug-fa4.pt "${common_args[@]}" || fa4_status=$?
test -s /tmp/fa4-grug-fa4.pt
.venv/bin/python experiments/fa4/compare_grug.py /tmp/fa4-grug-fa2.pt /tmp/fa4-grug-fa4.pt
echo "DIAGNOSTIC_ARM_EXIT_CODES fa2=$fa2_status fa4=$fa4_status"
test "$fa2_status" -eq 0 && test "$fa4_status" -eq 0
