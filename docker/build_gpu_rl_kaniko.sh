#!/usr/bin/env bash
# Build the GPU-RL image inside a disposable Iris Ubuntu task.
#
# kaniko builds for the architecture it runs on, so the image architecture is
# decided by where the Iris job lands, not by a flag. An amd64 build host on
# cw-us-east-02a with docker/Dockerfile.gpu-rl produces the linux/amd64 image; an
# aarch64 GB200 host on cw-us-east-08a with docker/Dockerfile.gpu-rl-arm64
# produces the linux/arm64 one. Everything below that depends on the host
# architecture — the crane release asset, the platform crane selects out of the
# multi-arch kaniko manifest, and the wheel MANIFEST platform tag — is derived
# from `uname -m` rather than hardcoded.
set -euo pipefail

: "${GITSHA:?}"
: "${DOCKER_USER_ID:?}"
: "${GHCR_TOKEN:?}"

# ARCH_TAG_SUFFIX keeps the two architectures apart in the registry. Every tag is
# derived from the git sha and the same commit builds both images, so without a
# suffix an arm64 build overwrites the amd64 tag of the identical sha — including
# the wheels tag, whose wheels are not interchangeable. The kaniko cache repo is
# split for the same reason. Both follow the build host rather than an operator
# env var, because a forgotten suffix is silent and destroys a shipped tag.
#
# The published prebuilt wheelhouse artifact holds linux_x86_64 wheels, so aarch64
# has no wheel source but the wheel-builder stage.
BUILD_ARCH=$(uname -m)
case "$BUILD_ARCH" in
  x86_64)
    CRANE_ASSET_ARCH=x86_64; KANIKO_PLATFORM=linux/amd64; WHEEL_PLATFORM_TAG=linux_x86_64
    ARCH_TAG_SUFFIX=""; DEFAULT_WHEEL_SOURCE=prebuilt-wheelhouse ;;
  aarch64)
    CRANE_ASSET_ARCH=arm64;  KANIKO_PLATFORM=linux/arm64; WHEEL_PLATFORM_TAG=linux_aarch64
    ARCH_TAG_SUFFIX="-arm64"; DEFAULT_WHEEL_SOURCE=wheel-builder ;;
  *) echo "unsupported build host architecture: $BUILD_ARCH" >&2; exit 2 ;;
esac
echo "[arch] build host=$BUILD_ARCH kaniko=$KANIKO_PLATFORM wheels=$WHEEL_PLATFORM_TAG tag-suffix=${ARCH_TAG_SUFFIX:-none}"

# Registry home is this repo's org, marin-community/MarinSkyRL. Declared here so a
# build pushes where it says it pushes without an ad-hoc env var at every call site.
GHCR_IMAGE_REPOSITORY="${GHCR_IMAGE_REPOSITORY:-ghcr.io/marin-community/marinskyrl}"

WHEEL_SOURCE="${WHEEL_SOURCE:-$DEFAULT_WHEEL_SOURCE}"
INSTALL_MEGATRON="${INSTALL_MEGATRON:-0}"
TAG_PREFIX="${TAG_PREFIX:-gpu-rl}"
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

# Baked pins are DECLARED in the Dockerfile and read from there — the build never
# carries a second copy of a version it might bake. A duplicate default in this
# script once drifted a full harbor release behind the Dockerfile, and every build
# stayed correct only because operators happened to pass an override; a build run
# without it would have silently shipped older harbor while reporting no change.
#
# An env override is now a hard error rather than a silent divergence. To change a
# pin, edit the Dockerfile and commit it, so the image always matches the source
# that claims to describe it.
PINNED_ARGS=(HARBOR_COMMIT VLLM_FORK_COMMIT VLLM_NATIVE_DONOR_COMMIT FLASH_ATTN_VERSION TORCH_VERSION)
for _arg in "${PINNED_ARGS[@]}"; do
  _declared="$(dockerfile_arg "$_arg")"
  if [ -z "$_declared" ]; then
    echo "ERROR: $DOCKERFILE declares no default for $_arg." >&2
    echo "Every baked pin must be declared in the Dockerfile so the build is reproducible from source." >&2
    exit 2
  fi
  _supplied="$(eval "printf '%s' \"\${$_arg:-}\"")"
  if [ -n "$_supplied" ] && [ "$_supplied" != "$_declared" ]; then
    echo "ERROR: $_arg was overridden to '$_supplied' but $DOCKERFILE declares '$_declared'." >&2
    echo "Baked pins are changed by editing and committing the Dockerfile, never by an env override," >&2
    echo "so that a built image always matches the committed source. Refusing to build." >&2
    exit 2
  fi
  eval "$_arg=\$_declared"
  echo "[pin] $_arg=$_declared (from $DOCKERFILE)"
done
unset _arg _declared _supplied

SNAPSHOT_FLAGS=()
if [ "${SINGLE_SNAPSHOT:-0}" = "1" ]; then
  SNAPSHOT_FLAGS=(--single-snapshot)
fi

if [ "${KANIKO_CACHE:-1}" = "0" ]; then
  CACHE_FLAGS=(--cache=false)
else
  KANIKO_CACHE_REPOSITORY="${KANIKO_CACHE_REPOSITORY:-${GHCR_IMAGE_REPOSITORY}/cache${ARCH_TAG_SUFFIX}}"
  CACHE_FLAGS=(--cache=true "--cache-repo=${KANIKO_CACHE_REPOSITORY}")
fi

IMAGE_TAG="${TAG_PREFIX}-${GITSHA}${ARCH_TAG_SUFFIX}"
DESTINATIONS=(--destination "${GHCR_IMAGE_REPOSITORY}:${IMAGE_TAG}")
if [ "${PUSH_FLOATING:-0}" = "1" ]; then
  DESTINATIONS+=(--destination "${GHCR_IMAGE_REPOSITORY}:${TAG_PREFIX}${ARCH_TAG_SUFFIX}")
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
    "CUDA=12.9 PY=cp312 PLATFORM=${WHEEL_PLATFORM_TAG}" \
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
  "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_${CRANE_ASSET_ARCH}.tar.gz" \
  -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane
# The kaniko executor tag is a multi-arch manifest, and crane defaults to
# linux/amd64 regardless of the host, so the platform has to be explicit or an
# aarch64 builder unpacks amd64 binaries it cannot run.
crane export --platform "$KANIKO_PLATFORM" gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true
test -x /kaniko/executor

export DOCKER_CONFIG=/kaniko/.docker
install -d -m 0700 "$DOCKER_CONFIG"
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$GHCR_TOKEN" | base64 | tr -d '\n')
printf '{"auths":{"ghcr.io":{"auth":"%s"}}}\n' "$AUTH" \
  > "$DOCKER_CONFIG/config.json"
chmod 0600 "$DOCKER_CONFIG/config.json"
unset AUTH GHCR_TOKEN
set -x

# When we pay the nvcc compile, keep the wheels. The wheel-builder stage is pushed as
# its own tag first, so the compiled vLLM-fork + flash-attn wheels survive as an image
# and can be turned into a prebuilt-wheelhouse artifact later without recompiling. The
# full build that follows reuses these layers from the cache, so this is one compile,
# not two. Skipped on the prebuilt path, where there is nothing new to preserve.
if [ "$WHEEL_SOURCE" = "wheel-builder" ] && [ "${PRESERVE_WHEELS:-1}" = "1" ]; then
  /kaniko/executor \
    --context "dir://${DOCKER_CONTEXT}" \
    --dockerfile "$DOCKERFILE" \
    --target wheel-builder \
    --build-arg WHEEL_SOURCE="$WHEEL_SOURCE" \
    --build-arg INSTALL_MEGATRON="$INSTALL_MEGATRON" \
    --build-arg GITSHA="$GITSHA" \
    --build-arg HARBOR_COMMIT="$HARBOR_COMMIT" \
    --build-arg VLLM_NATIVE_DONOR_ARCHIVE_SHA256="$NATIVE_ARCHIVE_SHA256" \
    --skip-unused-stages \
    --compressed-caching=false \
    "${CACHE_FLAGS[@]}" \
    --destination "${GHCR_IMAGE_REPOSITORY}:wheels-${GITSHA}${ARCH_TAG_SUFFIX}"
  echo "preserved wheel-builder stage as ${GHCR_IMAGE_REPOSITORY}:wheels-${GITSHA}${ARCH_TAG_SUFFIX}"
fi

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
