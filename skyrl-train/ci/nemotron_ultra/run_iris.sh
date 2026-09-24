#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --short)" ]]; then
    echo "Nemotron Ultra acceptance must run from a clean, committed worktree." >&2
    exit 2
fi

launch_config="${LAUNCH_CONFIG:?LAUNCH_CONFIG must name a complete SkyRL launch YAML}"
log_path="${LOG_PATH:-nemotron-ultra-rlvr-acceptance.log}"

PYTHONPATH=. uv run --frozen marinskyrl iris launch --config "$launch_config" 2>&1 | tee "$log_path"
PYTHONPATH=skyrl-train:. uv run --frozen python -m ci.nemotron_ultra.gate --log "$log_path"
