#!/usr/bin/env bash
set -euo pipefail

: "${PIVOT_RUN_ROOT:?Set the durable run URI}"
: "${PIVOT_TMP_ROOT:?Set the lifecycle-managed temporary URI}"
: "${RUN_ID:?Set a unique run ID}"
CONFIG_NAME="${1:-pivot_swe_smoke}"

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_ENV="${RUNTIME_ENV:-$REPOSITORY_ROOT/.iris-pivot-fsdp}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$RUNTIME_ENV" production fsdp

cd "$REPOSITORY_ROOT/skyrl-train"
rm -rf skyrl-gym
cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SKYRL_HOME="$REPOSITORY_ROOT"
export VLLM_USE_DEEP_GEMM=0

LOG_PATH="${IRIS_OUTPUT_DIR:-/tmp}/pivot-swe-smoke.log"
"$PYTHON" -m skyrl_train.entrypoints.pivot_swe_smoke --config-name "$CONFIG_NAME" \
  2>&1 | tee "$LOG_PATH"
