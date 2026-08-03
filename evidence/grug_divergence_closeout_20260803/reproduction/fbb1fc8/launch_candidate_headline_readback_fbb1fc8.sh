#!/usr/bin/env bash
set -euo pipefail

PRODUCT_ROOT=/home/romain/dev/marin-wt/grug-moe-execution-20260731
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
EXPECTED_SOURCE=fbb1fc8378601e0346d00d186809f10d1ad0360d
IMAGE='ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770'
PREFLIGHT_RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-fbb1fc8/preflight-paired-s1.json
HEADLINE_RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-fbb1fc8/headline-paired-s1.json
PREFLIGHT_READER="$PRODUCT_ROOT/.agents/tmp/verify_candidate_preflight_artifact.py"
HEADLINE_READER="$PRODUCT_ROOT/.agents/tmp/verify_candidate_headline_artifact.py"
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG=${IRIS_CONFIG:-"$MARIN_ROOT/lib/iris/config/cw-us-east-02a.yaml"}

test "$(git -C "$PRODUCT_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE"
git -C "$PRODUCT_ROOT" diff --quiet
git -C "$PRODUCT_ROOT" diff --cached --quiet
python -m py_compile "$PREFLIGHT_READER" "$HEADLINE_READER"
PREFLIGHT_READER_B64=$(base64 < "$PREFLIGHT_READER" | tr -d '\n')
HEADLINE_READER_B64=$(base64 < "$HEADLINE_READER" | tr -d '\n')

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
  --job-name grug-candidate-headline-readback-fbb1fc8-s1-20260803 \
  --task-image "$IMAGE" \
  -e PREFLIGHT_READER_B64 "$PREFLIGHT_READER_B64" \
  -e HEADLINE_READER_B64 "$HEADLINE_READER_B64" \
  -e PREFLIGHT_RESULT "$PREFLIGHT_RESULT" \
  -e HEADLINE_RESULT "$HEADLINE_RESULT" \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash -c 'set -euo pipefail; echo "$PREFLIGHT_READER_B64" | base64 -d > /tmp/verify-preflight.py; echo "$HEADLINE_READER_B64" | base64 -d > /tmp/verify-headline.py; /opt/openthoughts/envs/rl/bin/python /tmp/verify-preflight.py "$PREFLIGHT_RESULT" /tmp/preflight.raw.json /tmp/preflight.summary.json; exec /opt/openthoughts/envs/rl/bin/python /tmp/verify-headline.py "$HEADLINE_RESULT" /tmp/headline.raw.json /tmp/headline.summary.json'
