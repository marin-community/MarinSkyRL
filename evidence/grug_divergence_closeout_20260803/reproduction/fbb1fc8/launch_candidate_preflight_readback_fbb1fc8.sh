#!/usr/bin/env bash
set -euo pipefail

PRODUCT_ROOT=/home/romain/dev/marin-wt/grug-moe-execution-20260731
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
EXPECTED_SOURCE=fbb1fc8378601e0346d00d186809f10d1ad0360d
IMAGE='ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770'
RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-fbb1fc8/preflight-paired-s1.json
READER="$PRODUCT_ROOT/.agents/tmp/verify_candidate_preflight_artifact.py"
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG=${IRIS_CONFIG:-"$MARIN_ROOT/lib/iris/config/cw-us-east-02a.yaml"}

test "$(git -C "$PRODUCT_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE"
git -C "$PRODUCT_ROOT" diff --quiet
git -C "$PRODUCT_ROOT" diff --cached --quiet
python -m py_compile "$READER"
READER_B64=$(base64 < "$READER" | tr -d '\n')

cd "$MARIN_ROOT"
"$IRIS_BIN" --config "$IRIS_CONFIG" job run \
  --cpu 8 \
  --memory 32GB \
  --disk 100GB \
  --enable-extra-resources \
  --priority interactive \
  --max-retries 0 \
  --timeout 1800 \
  --no-wait \
  --no-sync \
  --job-name grug-candidate-preflight-readback-fbb1fc8-s1-20260803 \
  --task-image "$IMAGE" \
  -e READER_B64 "$READER_B64" \
  -e RESULT "$RESULT" \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash -c 'set -euo pipefail; echo "$READER_B64" | base64 -d > /tmp/verify-preflight.py; exec /opt/openthoughts/envs/rl/bin/python /tmp/verify-preflight.py "$RESULT" /tmp/preflight.raw.json /tmp/preflight.summary.json'
