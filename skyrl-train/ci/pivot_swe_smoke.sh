#!/usr/bin/env bash
set -euo pipefail

: "${PIVOT_RUN_ROOT:?Set the durable run URI}"
: "${PIVOT_TMP_ROOT:?Set the lifecycle-managed checkpoint URI}"
: "${RUN_ID:?Set a unique run ID}"

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
"$PYTHON" -m skyrl_train.entrypoints.pivot_swe_smoke \
  2>&1 | tee "$LOG_PATH"

"$PYTHON" -m skyrl_train.entrypoints.checkpoint_export \
  --config-name pivot_swe_smoke \
  checkpoint_export.step=1 \
  "checkpoint_export.checkpoint_path=\"${PIVOT_TMP_ROOT}/checkpoints/global_step_1\"" \
  "checkpoint_export.export_root=${PIVOT_RUN_ROOT}/exports" \
  2>&1 | tee -a "$LOG_PATH"
