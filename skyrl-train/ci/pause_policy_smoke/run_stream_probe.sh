#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "$0")/../../.." && pwd)"
probe_env="$repository_root/.iris-pause-probe-env"
source "$repository_root/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$repository_root" "$probe_env" development megatron

cd "$repository_root/skyrl-train"
rm -rf skyrl-gym
cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$repository_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_DEEP_GEMM=0

"$PYTHON" -m pytest -s tests/gpu/gpu_ci/test_pause_and_continue_generation.py \
  -m vllm -k streaming_chat_completion_crosses_weight_sync
