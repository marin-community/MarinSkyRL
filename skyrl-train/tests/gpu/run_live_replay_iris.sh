#!/usr/bin/env bash

# Run the real vLLM -> Megatron router replay check in an Iris GPU task.
# The frozen SkyRL bootstrap exposes the CUDA wheel's nvcc and libraries to
# FlashInfer, Ray workers, and Megatron native extensions.
set -euo pipefail

project_root="$(pwd)"
environment="$project_root/.venv"
activation_file="$project_root/.iris-runtime-env"

bash cloud/iris/bootstrap_runtime.sh "$project_root" "$environment" "$activation_file" megatron development
source "$activation_file"
python -m pytest -s -vv skyrl-train/tests/gpu/test_megatron_router_replay_live.py
