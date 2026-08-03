#!/usr/bin/env bash
set -euo pipefail

MSRL_ROOT=/home/romain/dev/marin-wt/grug-moe-execution-20260731
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
EXPECTED_SHA=fbb1fc8378601e0346d00d186809f10d1ad0360d
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"

cd "$MSRL_ROOT"
MEASUREMENT_SHA=$(git rev-parse HEAD)
test "$MEASUREMENT_SHA" = "$EXPECTED_SHA"
git diff --quiet
git diff --cached --quiet

DOCKER_AUTH_B64=$(jq -er '.auths["ghcr.io"].auth' /home/romain/.docker/config.json)
DOCKER_AUTH_RAW=$(printf '%s' "$DOCKER_AUTH_B64" | base64 -d)
TASK_DOCKER_USER=${DOCKER_AUTH_RAW%%:*}
TASK_GHCR_TOKEN=${DOCKER_AUTH_RAW#*:}
test -n "$TASK_DOCKER_USER"
test -n "$TASK_GHCR_TOKEN"
unset DOCKER_AUTH_B64 DOCKER_AUTH_RAW

SUBMIT_DIR=$(mktemp -d /tmp/grug-correctness-closeout-image-fbb1fc8.XXXXXX)
trap 'rm -rf -- "$SUBMIT_DIR"' EXIT
git archive HEAD | tar -x -C "$SUBMIT_DIR"
mkdir "$SUBMIT_DIR/config"
cp "$MARIN_ROOT"/config/*.yaml "$SUBMIT_DIR/config/"
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
  --job-name grug-fp32-combine-gpu-rl-amd64-fbb1fc8-20260803 \
  --task-image docker.io/library/ubuntu:22.04 \
  -e BUILD_B64 "$BUILD_SCRIPT_B64" \
  -e GITSHA "$MEASUREMENT_SHA" \
  -e DOCKER_USER_ID "$TASK_DOCKER_USER" \
  -e GHCR_TOKEN "$TASK_GHCR_TOKEN" \
  -e WHEEL_SOURCE wheel-builder \
  -- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'

unset TASK_GHCR_TOKEN
