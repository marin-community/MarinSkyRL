#!/usr/bin/env bash
# Build the linux/amd64 GPU-RL image inside a disposable Iris Ubuntu task.
set -euo pipefail

: "${GITSHA:?}"
: "${GHCR_IMAGE_REPOSITORY:?}"
: "${DOCKER_USER_ID:?}"
: "${GHCR_TOKEN:?}"

WHEEL_SOURCE="${WHEEL_SOURCE:-prebuilt-wheelhouse}"
INSTALL_MEGATRON="${INSTALL_MEGATRON:-0}"
TAG_PREFIX="${TAG_PREFIX:-gpu-rl}"
HARBOR_COMMIT="${HARBOR_COMMIT:-01c736a6}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.gpu-rl}"
DOCKER_CONTEXT=/app
DOCKERFILE_PATH="${DOCKER_CONTEXT}/${DOCKERFILE}"
WHEELHOUSE="${DOCKER_CONTEXT}/docker/wheelhouse"

if [ -z "${IRIS_TASK_ID:-}" ] || [ ! -f "$DOCKERFILE_PATH" ]; then
  echo "build_gpu_rl_kaniko.sh must run inside a disposable Iris task" >&2
  exit 2
fi

dockerfile_arg() {
  sed -n "s/^ARG $1=//p" "$DOCKERFILE_PATH" | head -n 1 | tr -d '"'
}

SNAPSHOT_FLAGS=()
if [ "${SINGLE_SNAPSHOT:-0}" = "1" ]; then
  SNAPSHOT_FLAGS=(--single-snapshot)
fi

if [ "${KANIKO_CACHE:-1}" = "0" ]; then
  CACHE_FLAGS=(--cache=false)
else
  : "${KANIKO_CACHE_REPOSITORY:?Required when KANIKO_CACHE=1}"
  CACHE_FLAGS=(--cache=true "--cache-repo=${KANIKO_CACHE_REPOSITORY}")
fi

DESTINATIONS=(--destination "${GHCR_IMAGE_REPOSITORY}:${TAG_PREFIX}-${GITSHA}")
if [ "${PUSH_FLOATING:-0}" = "1" ]; then
  DESTINATIONS+=(--destination "${GHCR_IMAGE_REPOSITORY}:${TAG_PREFIX}")
fi

APT_PACKAGES=(ca-certificates curl tar)
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  : "${PREBUILT_WHEEL_ARTIFACT_URI:?}"
  : "${PREBUILT_WHEEL_ARTIFACT_SHA256:?}"
  [[ "$PREBUILT_WHEEL_ARTIFACT_URI" == s3://* ]] || {
    echo "PREBUILT_WHEEL_ARTIFACT_URI must use s3://" >&2
    exit 2
  }
  [[ "$PREBUILT_WHEEL_ARTIFACT_SHA256" =~ ^[[:xdigit:]]{64}$ ]] || {
    echo "PREBUILT_WHEEL_ARTIFACT_SHA256 must be a SHA-256 digest" >&2
    exit 2
  }
  APT_PACKAGES+=(python3-pip)
elif [ "$WHEEL_SOURCE" != "wheel-builder" ]; then
  echo "unsupported WHEEL_SOURCE: $WHEEL_SOURCE" >&2
  exit 2
fi

apt-get update -y
apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"

NATIVE_ARCHIVE_SHA256=not-applicable
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  python3 -m pip install --no-cache-dir fsspec==2026.4.0 s3fs==2026.4.0
  python3 - "$PREBUILT_WHEEL_ARTIFACT_URI" /tmp/vllm-wheels.tar.gz <<'PY'
import shutil
import sys

import fsspec

with fsspec.open(sys.argv[1], "rb") as source, open(sys.argv[2], "wb") as output:
    shutil.copyfileobj(source, output)
PY
  NATIVE_ARCHIVE_SHA256=$(sha256sum /tmp/vllm-wheels.tar.gz | cut -d ' ' -f 1)
  if [ "${NATIVE_ARCHIVE_SHA256,,}" != "${PREBUILT_WHEEL_ARTIFACT_SHA256,,}" ]; then
    echo "wheel artifact SHA-256 mismatch" >&2
    exit 1
  fi

  ARTIFACT_DIR=$(mktemp -d /tmp/vllm-wheel-artifact.XXXXXX)
  tar -xzf /tmp/vllm-wheels.tar.gz -C "$ARTIFACT_DIR"
  ARTIFACT_WHEELS="${ARTIFACT_DIR}/wheels"

  printf '%s\n' \
    "VLLM_FORK_COMMIT=$(dockerfile_arg VLLM_NATIVE_DONOR_COMMIT)" \
    "FLASH_ATTN_VERSION=$(dockerfile_arg FLASH_ATTN_VERSION)" \
    "TORCH_VERSION=$(dockerfile_arg TORCH_VERSION)" \
    "TORCH_CUDA_ARCH_LIST=$(dockerfile_arg TORCH_CUDA_ARCH_LIST)" \
    "CUDA=12.8 PY=cp312 PLATFORM=linux_x86_64" \
    > /tmp/expected-wheel-manifest
  cmp /tmp/expected-wheel-manifest "$ARTIFACT_WHEELS/MANIFEST"
  test "$(find "$ARTIFACT_WHEELS" -maxdepth 1 -type f -name 'vllm-*.whl' | wc -l)" -eq 1
  test "$(find "$ARTIFACT_WHEELS" -maxdepth 1 -type f -name 'flash_attn-*.whl' | wc -l)" -eq 1

  mkdir -p "$WHEELHOUSE"
  find "$WHEELHOUSE" -maxdepth 1 -type f \
    \( -name '*.whl' -o -name MANIFEST \) -delete
  install -m 0644 "$ARTIFACT_WHEELS"/MANIFEST \
    "$ARTIFACT_WHEELS"/vllm-*.whl \
    "$ARTIFACT_WHEELS"/flash_attn-*.whl \
    "$WHEELHOUSE/"
  echo "validated and staged prebuilt vLLM wheelhouse"
fi
unset PREBUILT_WHEEL_ARTIFACT_URI PREBUILT_WHEEL_ARTIFACT_SHA256

cd /tmp
CRANE_VERSION=v0.20.2
curl -fsSL \
  "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz" \
  -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane
crane export gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true

export DOCKER_CONFIG=/kaniko/.docker
install -d -m 0700 "$DOCKER_CONFIG"
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$GHCR_TOKEN" | base64 | tr -d '\n')
printf '{"auths":{"ghcr.io":{"auth":"%s"}}}\n' "$AUTH" \
  > "$DOCKER_CONFIG/config.json"
chmod 0600 "$DOCKER_CONFIG/config.json"
unset AUTH GHCR_TOKEN
set -x

exec /kaniko/executor \
  --context "dir://${DOCKER_CONTEXT}" \
  --dockerfile "$DOCKERFILE" \
  --build-arg WHEEL_SOURCE="$WHEEL_SOURCE" \
  --build-arg INSTALL_MEGATRON="$INSTALL_MEGATRON" \
  --build-arg GITSHA="$GITSHA" \
  --build-arg HARBOR_COMMIT="$HARBOR_COMMIT" \
  --build-arg VLLM_NATIVE_DONOR_ARCHIVE_SHA256="$NATIVE_ARCHIVE_SHA256" \
  --skip-unused-stages \
  "${SNAPSHOT_FLAGS[@]}" \
  --compressed-caching=false \
  "${CACHE_FLAGS[@]}" \
  "${DESTINATIONS[@]}"
