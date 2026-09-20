#!/usr/bin/env bash
set -euo pipefail

# Run the same Grug probe under the two experimental arms and the immutable
# pre-refresh Marin baseline. Each arm uses its own frozen project environment.
world_size="${1:?usage: run_three_arm_h100.sh WORLD_SIZE SHAPE [STEPS]}"
shape="${2:?usage: run_three_arm_h100.sh WORLD_SIZE SHAPE [STEPS]}"
steps="${3:-1}"
baseline_commit=4d798b12c9545c73ab893883c890633ee1db353e
baseline_dir=/tmp/fa4-marin-baseline

bash experiments/fa4/run_grug_h100.sh "$world_size" "$shape" "$steps"

git init --quiet "$baseline_dir"
git -C "$baseline_dir" remote add origin https://github.com/marin-community/MarinSkyRL.git
git -C "$baseline_dir" fetch --depth 1 origin "$baseline_commit"
git -C "$baseline_dir" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$baseline_dir" rev-parse HEAD)" = "$baseline_commit"
# Iris sets UV_PROJECT_ENVIRONMENT=/app/.venv; isolate the baseline explicitly.
env "UV_PROJECT_ENVIRONMENT=$baseline_dir/.venv" uv sync --project "$baseline_dir" \
    --frozen --extra vllm --extra megatron --group dev

baseline_env=(
    "PYTHONPATH=$baseline_dir/skyrl-train"
    NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1
    NVTE_FLASH_ATTN_V4=0 RAY_DEDUP_LOGS=0
)
env "${baseline_env[@]}" timeout 900 "$baseline_dir/.venv/bin/python" \
    experiments/fa4/grug_step.py --output /tmp/fa4-grug-old-fa2.pt \
    --world-size "$world_size" --shape "$shape" --steps "$steps" --cp-comm-type default
.venv/bin/python experiments/fa4/compare_grug.py /tmp/fa4-grug-old-fa2.pt /tmp/fa4-grug-fa2.pt
.venv/bin/python experiments/fa4/compare_grug.py /tmp/fa4-grug-old-fa2.pt /tmp/fa4-grug-fa4.pt
