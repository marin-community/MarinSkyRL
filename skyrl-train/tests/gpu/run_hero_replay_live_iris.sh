#!/usr/bin/env bash

# Run the actual Hero vLLM -> Megatron route-replay test in Iris.
set -euo pipefail

project_root="$(pwd)"
environment="$project_root/.venv"
activation_file="$project_root/.iris-runtime-env"

bash cloud/iris/bootstrap_runtime.sh "$project_root" "$environment" "$activation_file" megatron development
source "$activation_file"
python -m pytest -s -vv skyrl-train/tests/gpu/test_hero_router_replay_live.py
