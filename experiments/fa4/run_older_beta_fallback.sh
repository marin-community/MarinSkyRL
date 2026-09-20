#!/usr/bin/env bash
set -euo pipefail

# Disposable version-isolation probe. The project lock is not modified;
# only this task environment substitutes the selected beta wheel.
beta="${1:-28}"
case "$beta" in
    28 | 29 | 30) ;;
    *) echo "unsupported beta: $beta" >&2; exit 2 ;;
esac
uv sync --frozen --extra vllm --extra megatron --extra fa4 --group dev
common_env=(PYTHONPATH=skyrl-train NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 RAY_DEDUP_LOGS=0)
common_args=(--world-size 1 --shape toy --steps 1 --capture-grads)
env "${common_env[@]}" NVTE_FLASH_ATTN_V4=0 timeout 900 .venv/bin/python \
    experiments/fa4/grug_step.py --output /tmp/fa4-beta28-fa2.pt "${common_args[@]}"

uv pip install --python .venv/bin/python --no-deps \
    "flash-attn-4 @ https://github.com/Dao-AILab/flash-attention/releases/download/fa4-v4.0.0.beta${beta}/flash_attn_4-4.0.0b${beta}-py3-none-any.whl"
.venv/bin/python -c 'from importlib.metadata import version; import sys; assert version("flash-attn-4") == "4.0.0b" + sys.argv[1]' "$beta"
env "${common_env[@]}" NVTE_FLASH_ATTN_V4=1 timeout 900 .venv/bin/python \
    experiments/fa4/grug_step.py --output /tmp/fa4-beta28-fa4.pt "${common_args[@]}"
.venv/bin/python experiments/fa4/compare_grug.py /tmp/fa4-beta28-fa2.pt /tmp/fa4-beta28-fa4.pt
