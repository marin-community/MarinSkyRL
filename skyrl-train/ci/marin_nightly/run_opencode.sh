#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

launch_config="${LAUNCH_CONFIG:?LAUNCH_CONFIG must name a complete SkyRL launch YAML}"
log_path="${LOG_PATH:-opencode-nightly.log}"
mode="${OPENCODE_MODE:-nightly}"
started_at="$(date +%s)"

case "$mode" in
    nightly)
        gate_spec="skyrl-train/ci/marin_nightly/specs/opencode-qwen3-8b.json"
        ;;
    compaction-stress)
        gate_spec="skyrl-train/ci/marin_nightly/specs/opencode-compaction-stress.json"
        ;;
    overflow-stress)
        gate_spec="skyrl-train/ci/marin_nightly/specs/opencode-overflow-stress.json"
        ;;
    *)
        echo "unknown OPENCODE_MODE: $mode" >&2
        exit 2
        ;;
esac

PYTHONPATH=. uv run --frozen marinskyrl iris launch --config "$launch_config" 2>&1 | tee "$log_path"

finished_at="$(date +%s)"
uv run --frozen python skyrl-train/ci/marin_nightly/gate.py \
    --log "$log_path" \
    --spec "$gate_spec" \
    --wall-clock-seconds "$((finished_at - started_at))"
