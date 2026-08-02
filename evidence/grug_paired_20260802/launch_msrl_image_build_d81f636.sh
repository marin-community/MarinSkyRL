#!/usr/bin/env bash
set -euo pipefail

MSRL_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
EXPECTED_SHA=d81f6364ab66947cdf520a3a42a274b586e830da
IRIS_BIN=/home/romain/dev/marin/.venv/bin/iris

cd "$MSRL_ROOT"
MEASUREMENT_SHA=$(git rev-parse HEAD)
test "$MEASUREMENT_SHA" = "$EXPECTED_SHA"
test -z "$(git status --porcelain)"

DOCKER_AUTH_B64=$(jq -er '.auths["ghcr.io"].auth' /home/romain/.docker/config.json)
DOCKER_AUTH_RAW=$(printf '%s' "$DOCKER_AUTH_B64" | base64 -d)
TASK_DOCKER_USER=${DOCKER_AUTH_RAW%%:*}
TASK_GHCR_TOKEN=${DOCKER_AUTH_RAW#*:}
test -n "$TASK_DOCKER_USER"
test -n "$TASK_GHCR_TOKEN"
unset DOCKER_AUTH_B64 DOCKER_AUTH_RAW

SUBMIT_DIR=$(mktemp -d /tmp/grug-closeout/msrl-submit-d81f636.XXXXXX)
trap 'rm -rf -- "$SUBMIT_DIR"' EXIT
git archive HEAD | tar -x -C "$SUBMIT_DIR"
mkdir "$SUBMIT_DIR/config"
cp /home/romain/dev/marin/config/*.yaml "$SUBMIT_DIR/config/"
cd "$SUBMIT_DIR"
BUILD_SCRIPT_B64=$(base64 < docker/build_gpu_rl_kaniko.sh | tr -d '\n')

"$IRIS_BIN" --cluster cw-rno2a job run \
  --enable-extra-resources \
  --cpu 48 \
  --memory 512GB \
  --disk 400GB \
  --priority interactive \
  --max-retries 0 \
  --timeout 18000 \
  --no-wait \
  --no-sync \
  --job-name grug-paired-gpu-rl-amd64-d81f636-r2-20260802 \
  --task-image docker.io/library/ubuntu:22.04 \
  -e BUILD_B64 "$BUILD_SCRIPT_B64" \
  -e GITSHA "$MEASUREMENT_SHA" \
  -e DOCKER_USER_ID "$TASK_DOCKER_USER" \
  -e GHCR_TOKEN "$TASK_GHCR_TOKEN" \
  -e WHEEL_SOURCE wheel-builder \
  -- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'

unset TASK_GHCR_TOKEN
