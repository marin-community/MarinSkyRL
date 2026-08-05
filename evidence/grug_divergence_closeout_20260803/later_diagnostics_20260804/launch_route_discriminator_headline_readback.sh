#!/usr/bin/env bash
set -euo pipefail

EVIDENCE_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
IMAGE=ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770
RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/headline-discriminator-s1-rno-cpu8-mem768.json
READER="$EVIDENCE_ROOT/.agents/tmp/verify_route_discriminator_headline.py"
READER_SHA=ea32f61a06c8e75d93cacf2c93a057e9452eb9c5d181130faff8c596ddfeea6b
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG=${IRIS_CONFIG:-"$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"}

python -m py_compile "$READER"
test "$(sha256sum "$READER" | cut -d' ' -f1)" = "$READER_SHA"
READER_B64=$(base64 < "$READER" | tr -d '\n')

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY_RUN reader verified: result=%s reader_sha=%s cpu=1 config=%s\n' \
    "$RESULT" "$READER_SHA" "$IRIS_CONFIG"
  exit 0
fi

cd "$MARIN_ROOT"
# shellcheck disable=SC2016  # expanded inside the task container
"$IRIS_BIN" --config "$IRIS_CONFIG" job run \
  --cpu 1 \
  --memory 32GB \
  --disk 100GB \
  --enable-extra-resources \
  --priority interactive \
  --max-retries 0 \
  --timeout 1800 \
  --no-wait \
  --no-sync \
  --job-name grug-route-discriminator-headline-readback-s1-rno-cpu8-mem768-20260804 \
  --task-image "$IMAGE" \
  -e READER_B64 "$READER_B64" \
  -e RESULT "$RESULT" \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash -c 'set -euo pipefail; echo "$READER_B64" | base64 -d > /tmp/verify-headline.py; exec /opt/openthoughts/envs/rl/bin/python /tmp/verify-headline.py "$RESULT"'
