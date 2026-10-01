#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RUNTIME_COMMIT="$(bash "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" --commit "$REPOSITORY_ROOT")"
MARIN_ROOT="${RUNNER_TEMP:-$(mktemp -d)}/cat-count-marin"
MARIN_REVISION="${MARIN_REVISION:-main}"
git clone --filter=blob:none --no-checkout https://github.com/marin-community/marin.git "$MARIN_ROOT"
git -C "$MARIN_ROOT" fetch origin "$MARIN_REVISION"
git -C "$MARIN_ROOT" checkout --detach FETCH_HEAD
echo "CAT_COUNT_NIGHTLY marin=$(git -C "$MARIN_ROOT" rev-parse HEAD) runtime=$RUNTIME_COMMIT"

export PYTHONPATH="$REPOSITORY_ROOT/skyrl-train${PYTHONPATH:+:$PYTHONPATH}"
uv run --project "$MARIN_ROOT" --frozen --extra cpu --no-group dev \
  python "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/cat_count_nightly.py" \
  --marin-root "$MARIN_ROOT" --runtime-commit "$RUNTIME_COMMIT" \
  --cluster "${TARGET_CLUSTER:-cw-rno2a}" \
  --job-name "${JOB_NAME:?JOB_NAME is required}" \
  --log "${LOG:-cat-count-nightly.log}" \
  --spec "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/specs/cat-count-canary-qwen2.5-0.5b-async.json"
