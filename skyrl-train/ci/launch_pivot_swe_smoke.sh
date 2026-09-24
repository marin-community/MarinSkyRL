#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ID="pivot-swe-smoke-$(date -u +%Y%m%d-%H%M%S)"
STORAGE_USER="${USER:?Set the storage user}"
PIVOT_RUN_ROOT="s3://marin-us-east-02a/marin/users/${STORAGE_USER}/skyrl/${RUN_ID}"
PIVOT_TMP_ROOT="s3://marin-us-east-02a/tmp/ttl=1d/skyrl/users/${STORAGE_USER}/${RUN_ID}"

cd "$REPOSITORY_ROOT"
PYTHONPATH="$REPOSITORY_ROOT" "$REPOSITORY_ROOT/.venv/bin/python" -c '
import shutil
import subprocess
from pathlib import Path
from cloud.iris.runtime_bundle import BUNDLE_IDENTITY_FILE, build_runtime_bundle
commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
workspace = build_runtime_bundle(commit)
shutil.copy2(workspace / BUNDLE_IDENTITY_FILE, Path(BUNDLE_IDENTITY_FILE))
'

echo "Predictions and errors: ${PIVOT_RUN_ROOT}/exports/dumped_evals"
echo "Paired report: ${PIVOT_RUN_ROOT}/diagnostics/comparison.jsonl"
echo "Training actions: ${PIVOT_RUN_ROOT}/diagnostics/training.jsonl"

"$REPOSITORY_ROOT/.venv/bin/iris" --cluster marin job run \
  --target-cluster cw-rno2a \
  --job-name "$RUN_ID" \
  --gpu H100x1 --enable-extra-resources \
  --cpu 8 --memory 64GB --disk 200GB \
  --timeout 3600 --no-sync \
  -e RUN_ID "$RUN_ID" \
  -e PIVOT_RUN_ROOT "$PIVOT_RUN_ROOT" \
  -e PIVOT_TMP_ROOT "$PIVOT_TMP_ROOT" \
  -- bash skyrl-train/ci/pivot_swe_smoke.sh
