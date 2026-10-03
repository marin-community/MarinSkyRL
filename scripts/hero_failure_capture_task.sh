#!/usr/bin/env bash
set -euo pipefail

export SKYRL_HOME="$PWD"
export PYTHONPATH="$PWD/skyrl-train:$PWD/skyrl-gym:$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Custom Hero tasks must publish the same root used by their Ray workers.
export SKYRL_DEBUG_ARTIFACT_DIR="${SKYRL_DEBUG_ARTIFACT_DIR:-/tmp/debug}"
export OT_AGENT_RAY_LOG_SYNC_INTERVAL_S="${OT_AGENT_RAY_LOG_SYNC_INTERVAL_S:-60}"
exec python -m scripts.hero_failure_capture task "$@"
