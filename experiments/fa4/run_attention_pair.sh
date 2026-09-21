#!/usr/bin/env bash
set -euo pipefail

# Matched, separate-process TE attention probes. Use the same script on H100
# and GB200 so only hardware and NVTE_FLASH_ATTN_V4 vary between arms.
batch="${1:?usage: run_attention_pair.sh BATCH SEQ HEADS KV_HEADS HEAD_DIM LEFT [WARMUPS] [SAMPLES] [RIGHT]}"
seq="${2:?usage: run_attention_pair.sh BATCH SEQ HEADS KV_HEADS HEAD_DIM LEFT [WARMUPS] [SAMPLES] [RIGHT]}"
heads="${3:?usage: run_attention_pair.sh BATCH SEQ HEADS KV_HEADS HEAD_DIM LEFT [WARMUPS] [SAMPLES] [RIGHT]}"
kv_heads="${4:?usage: run_attention_pair.sh BATCH SEQ HEADS KV_HEADS HEAD_DIM LEFT [WARMUPS] [SAMPLES] [RIGHT]}"
head_dim="${5:?usage: run_attention_pair.sh BATCH SEQ HEADS KV_HEADS HEAD_DIM LEFT [WARMUPS] [SAMPLES] [RIGHT]}"
window="${6:?usage: run_attention_pair.sh BATCH SEQ HEADS KV_HEADS HEAD_DIM LEFT [WARMUPS] [SAMPLES] [RIGHT]}"
warmups="${7:-3}"
samples="${8:-10}"
window_right="${9:-0}"
reference="${10:-}"

if [[ -z "${FA4_RUNTIME_PYTHON:-}" ]]; then
    uv sync --frozen --extra vllm --extra megatron --group dev
    FA4_RUNTIME_PYTHON=.venv/bin/python
fi
output_dir="${IRIS_OUTPUT_DIR:-/tmp}"
common_env=(NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=0 NVTE_FLASH_ATTN=1 NVTE_FLASH_ATTN_V3=0)
common_args=(
    --batch "$batch" --seq "$seq" --heads "$heads" --kv-heads "$kv_heads"
    --head-dim "$head_dim" --window-left "$window" --window-right "$window_right"
    --warmups "$warmups" --samples "$samples"
)
if [[ "$reference" == "reference" ]]; then
    common_args+=(--reference-sdpa)
elif [[ -n "$reference" ]]; then
    echo "unknown reference mode: $reference" >&2
    exit 2
fi
env "${common_env[@]}" NVTE_FLASH_ATTN_V2=1 NVTE_FLASH_ATTN_V4=0 timeout 900 "$FA4_RUNTIME_PYTHON" \
    experiments/fa4/probe_attention.py --output "$output_dir/fa2-attention.pt" "${common_args[@]}"
env "${common_env[@]}" NVTE_FLASH_ATTN_V2=0 NVTE_FLASH_ATTN_V4=1 timeout 900 "$FA4_RUNTIME_PYTHON" \
    experiments/fa4/probe_attention.py --output "$output_dir/fa4-attention.pt" "${common_args[@]}"
"$FA4_RUNTIME_PYTHON" experiments/fa4/compare_attention.py "$output_dir/fa2-attention.pt" "$output_dir/fa4-attention.pt"
