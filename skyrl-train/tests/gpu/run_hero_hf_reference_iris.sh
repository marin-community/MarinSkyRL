#!/usr/bin/env bash

# Compare a trained replay rollout with the repository's plain BF16 Hero model.
set -euo pipefail

project_root="$(pwd)"
environment="$project_root/.venv"
activation_file="$project_root/.iris-runtime-env"

bash cloud/iris/bootstrap_runtime.sh "$project_root" "$environment" "$activation_file" megatron development
source "$activation_file"
python skyrl-train/tests/gpu/hero_trained_hf_reference.py
