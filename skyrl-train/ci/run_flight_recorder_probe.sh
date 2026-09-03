#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-frprobe-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$ENV_DIR" production
cd "$REPOSITORY_ROOT/skyrl-train"
rm -rf skyrl-gym && cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$PYTHON" ci/probe_flight_recorder.py
echo "::: PROBE EXIT=$?"
