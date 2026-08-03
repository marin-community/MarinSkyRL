#!/usr/bin/env bash
set -euo pipefail

MSRL_ROOT=/home/romain/dev/marin-wt/grug-moe-execution-20260731
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
EXPECTED_SHA=fbb1fc8378601e0346d00d186809f10d1ad0360d
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG="$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"
WHEEL_STAGE=ghcr.io/marin-community/marinskyrl@sha256:ac9373e686c6c378c8cd390a7bc54363fd9c6a9d6968f083c463bebf7c1ca24f
REUSE_DOCKERFILE=docker/Dockerfile.gpu-rl-reuse-fbb1fc8

cd "$MSRL_ROOT"
MEASUREMENT_SHA=$(git rev-parse HEAD)
test "$MEASUREMENT_SHA" = "$EXPECTED_SHA"
git diff --quiet
git diff --cached --quiet
test "$(sha256sum docker/Dockerfile.gpu-rl | cut -d' ' -f1)" = b11a8aa9aa63ac12cc044ea8245be382fd404d0c617c01f0a67c0a3d02d32f48
test "$(sha256sum docker/build_gpu_rl_kaniko.sh | cut -d' ' -f1)" = 121646154e082dc55bd8807e1741f46dd07fdee702bac3631689a93045fa6900

DOCKER_AUTH_B64=$(jq -er '.auths["ghcr.io"].auth' /home/romain/.docker/config.json)
DOCKER_AUTH_RAW=$(printf '%s' "$DOCKER_AUTH_B64" | base64 -d)
TASK_DOCKER_USER=${DOCKER_AUTH_RAW%%:*}
TASK_GHCR_TOKEN=${DOCKER_AUTH_RAW#*:}
test -n "$TASK_DOCKER_USER"
test -n "$TASK_GHCR_TOKEN"
unset DOCKER_AUTH_B64 DOCKER_AUTH_RAW

SUBMIT_DIR=$(mktemp -d /tmp/grug-correctness-closeout-image-reuse-fbb1fc8.XXXXXX)
trap 'rm -rf -- "$SUBMIT_DIR"' EXIT
git archive HEAD | tar -x -C "$SUBMIT_DIR"
mkdir "$SUBMIT_DIR/config"
cp "$MARIN_ROOT"/config/*.yaml "$SUBMIT_DIR/config/"
cp "$SUBMIT_DIR/docker/Dockerfile.gpu-rl" "$SUBMIT_DIR/$REUSE_DOCKERFILE"
sed -i \
  "s|^FROM \${WHEEL_SOURCE} AS wheels$|FROM $WHEEL_STAGE AS wheels|" \
  "$SUBMIT_DIR/$REUSE_DOCKERFILE"
test "$(sha256sum "$SUBMIT_DIR/$REUSE_DOCKERFILE" | cut -d' ' -f1)" = 5f918aa16bdd185d025220a1bf89943e2a4d9d721d5be968504299ce0f9e97dd
test "$(rg -c -F "FROM $WHEEL_STAGE AS wheels" "$SUBMIT_DIR/$REUSE_DOCKERFILE")" = 1
cd "$SUBMIT_DIR"
BUILD_SCRIPT_B64=$(base64 < docker/build_gpu_rl_kaniko.sh | tr -d '\n')

"$IRIS_BIN" --config "$IRIS_CONFIG" job run \
  --enable-extra-resources \
  --cpu 48 \
  --memory 512GB \
  --disk 400GB \
  --priority interactive \
  --max-retries 0 \
  --timeout 18000 \
  --no-wait \
  --no-sync \
  --job-name grug-fp32-combine-gpu-rl-amd64-fbb1fc8-reuse-v2-20260803 \
  --task-image docker.io/library/ubuntu:22.04 \
  -e BUILD_B64 "$BUILD_SCRIPT_B64" \
  -e GITSHA "$MEASUREMENT_SHA" \
  -e DOCKER_USER_ID "$TASK_DOCKER_USER" \
  -e GHCR_TOKEN "$TASK_GHCR_TOKEN" \
  -e WHEEL_SOURCE wheel-builder \
  -e PRESERVE_WHEELS 0 \
  -e DOCKERFILE "$REUSE_DOCKERFILE" \
  -- bash -lc 'echo "$BUILD_B64" | base64 -d > /tmp/build.sh && exec bash /tmp/build.sh'

unset TASK_GHCR_TOKEN
