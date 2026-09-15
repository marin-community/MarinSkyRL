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
# SHELLOPTS can carry xtrace across Bash processes, so disable it before reading
# registry credentials from the environment.
set +x
set -euo pipefail

: "${GITSHA:?}"
: "${REGISTRY_USER:?}"
: "${REGISTRY_TOKEN:?}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ARCH_TAG_SUFFIX keeps the two architectures apart in the registry. Every tag is
# derived from the git sha and the same commit builds both images, so without a
# suffix an arm64 build overwrites the amd64 tag of the identical sha — including
# the wheels tag, whose wheels are not interchangeable. The kaniko cache repo is
# split for the same reason. Both follow the build host rather than an operator
# env var, because a forgotten suffix is silent and destroys a shipped tag.
#
# The published prebuilt wheelhouse artifact holds linux_x86_64 wheels, so aarch64
# has no wheel source but the wheel-builder stage. The Dockerfile follows the host
# too: an aarch64 job that built Dockerfile.gpu-rl would bake a linux_x86_64 wheel
# MANIFEST and push it under the -arm64 tag.
BUILD_ARCH=$(uname -m)
case "$BUILD_ARCH" in
  x86_64)
    CRANE_ASSET_ARCH=x86_64; KANIKO_PLATFORM=linux/amd64; WHEEL_PLATFORM_TAG=linux_x86_64
    ARCH_TAG_SUFFIX=""; DEFAULT_WHEEL_SOURCE=prebuilt-wheelhouse
    DEFAULT_DOCKERFILE=docker/Dockerfile.gpu-rl ;;
  aarch64)
    CRANE_ASSET_ARCH=arm64;  KANIKO_PLATFORM=linux/arm64; WHEEL_PLATFORM_TAG=linux_aarch64
    ARCH_TAG_SUFFIX="-arm64"; DEFAULT_WHEEL_SOURCE=wheel-builder
    DEFAULT_DOCKERFILE=docker/Dockerfile.gpu-rl-arm64 ;;
  *) echo "unsupported build host architecture: $BUILD_ARCH" >&2; exit 2 ;;
esac
echo "[arch] build host=$BUILD_ARCH kaniko=$KANIKO_PLATFORM wheels=$WHEEL_PLATFORM_TAG tag-suffix=${ARCH_TAG_SUFFIX:-none}"

# Registry home is this repo's org, marin-community/MarinSkyRL. Declared here so a
# build pushes where it says it pushes without an ad-hoc env var at every call site.
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-ghcr.io/marin-community/marinskyrl}"
REGISTRY_HOST="${IMAGE_REPOSITORY%%/*}"
if [ "$REGISTRY_HOST" = "$IMAGE_REPOSITORY" ]; then
  echo "IMAGE_REPOSITORY must include a registry hostname and repository path" >&2
  exit 2
fi

WHEEL_SOURCE="${WHEEL_SOURCE:-$DEFAULT_WHEEL_SOURCE}"
HF_WHEEL_REPOSITORY="${HF_WHEEL_REPOSITORY:-}"
INSTALL_MEGATRON="${INSTALL_MEGATRON:-0}"
TAG_PREFIX="${TAG_PREFIX:-gpu-rl}"
DOCKERFILE="${DOCKERFILE:-$DEFAULT_DOCKERFILE}"
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

verify_wheelhouse() {
  local wheelhouse="$1"
  cmp /tmp/expected-wheel-manifest "$wheelhouse/MANIFEST"
  test "$(find "$wheelhouse" -maxdepth 1 -type f -name 'vllm-*.whl' | wc -l)" -eq 1
  test "$(find "$wheelhouse" -maxdepth 1 -type f -name 'flash_attn-*.whl' | wc -l)" -eq 1
  (cd "$wheelhouse" && sha256sum --check SHA256SUMS)
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
PINNED_ARGS=(HARBOR_COMMIT VLLM_FORK_COMMIT FLASH_ATTN_VERSION TORCH_VERSION)
# The native donor names whose compiled extensions a prebuilt or automatically
# resolved wheelhouse carries.
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ] || [ "$WHEEL_SOURCE" = "auto" ]; then
  PINNED_ARGS+=(VLLM_NATIVE_DONOR_COMMIT)
fi
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

EXPECTED_WHEEL_COMMIT="$VLLM_FORK_COMMIT"
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ] || [ "$WHEEL_SOURCE" = "auto" ]; then
  EXPECTED_WHEEL_COMMIT="$VLLM_NATIVE_DONOR_COMMIT"
fi
printf '%s\n' \
  "VLLM_FORK_COMMIT=${EXPECTED_WHEEL_COMMIT}" \
  "FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION}" \
  "TORCH_VERSION=${TORCH_VERSION}" \
  "TORCH_CUDA_ARCH_LIST=$(dockerfile_arg TORCH_CUDA_ARCH_LIST)" \
  "CUDA=12.9 PY=cp312 PLATFORM=${WHEEL_PLATFORM_TAG}" \
  > /tmp/expected-wheel-manifest
EXPECTED_WHEEL_MANIFEST_SHA256=$(sha256sum /tmp/expected-wheel-manifest | cut -d ' ' -f 1)

SNAPSHOT_FLAGS=()
if [ "${SINGLE_SNAPSHOT:-0}" = "1" ]; then
  SNAPSHOT_FLAGS=(--single-snapshot)
fi

if [ "${KANIKO_CACHE:-1}" = "0" ]; then
  CACHE_FLAGS=(--cache=false)
else
  KANIKO_CACHE_REPOSITORY="${KANIKO_CACHE_REPOSITORY:-${IMAGE_REPOSITORY}/cache${ARCH_TAG_SUFFIX}}"
  CACHE_FLAGS=(--cache=true "--cache-repo=${KANIKO_CACHE_REPOSITORY}")
fi

# Large cached layers traverse the CoreWeave-to-GHCR path for several minutes.
# Kaniko otherwise defaults every registry operation to zero retries, so one
# reset discards an otherwise healthy multi-hour build. Retry extraction,
# download, and push failures inside the same task while preserving its cache.
REGISTRY_RETRY_FLAGS=(
  --image-fs-extract-retry=3
  --image-download-retry=3
  --push-retry=3
)

IMAGE_TAG="${TAG_PREFIX}-${GITSHA}${ARCH_TAG_SUFFIX}"
DESTINATIONS=(--destination "${IMAGE_REPOSITORY}:${IMAGE_TAG}")
if [ "${PUSH_FLOATING:-0}" = "1" ]; then
  DESTINATIONS+=(--destination "${IMAGE_REPOSITORY}:${TAG_PREFIX}${ARCH_TAG_SUFFIX}")
fi

APT_PACKAGES=(ca-certificates curl tar)
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ] || [ "$WHEEL_SOURCE" = "auto" ] || [ -n "$HF_WHEEL_REPOSITORY" ]; then
  APT_PACKAGES+=(python3-pip)
fi

if [ "$WHEEL_SOURCE" = "auto" ]; then
  : "${HF_WHEEL_REPOSITORY:?WHEEL_SOURCE=auto requires HF_WHEEL_REPOSITORY}"
  : "${HF_TOKEN:?WHEEL_SOURCE=auto requires HF_TOKEN so a cache miss can be published}"
fi
if [ "$WHEEL_SOURCE" = "wheel-builder" ] && [ -n "$HF_WHEEL_REPOSITORY" ]; then
  : "${HF_TOKEN:?HF_WHEEL_REPOSITORY requires HF_TOKEN for source builds}"
fi

apt-get update -y
apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"

if [ "$WHEEL_SOURCE" = "auto" ]; then
  [ "$BUILD_ARCH" = x86_64 ] || { echo "automatic Hugging Face wheel reuse is currently amd64-only" >&2; exit 2; }
  HF_WHEEL_ROOT="https://huggingface.co/datasets/${HF_WHEEL_REPOSITORY}/resolve/main/wheelhouses/${WHEEL_PLATFORM_TAG}/${EXPECTED_WHEEL_MANIFEST_SHA256}"
  HF_DIGEST_FILE=/tmp/hf-vllm-wheels.tar.gz.sha256
  if ! HF_DIGEST_STATUS=$(curl -sS -L -o "$HF_DIGEST_FILE" -w '%{http_code}' "${HF_WHEEL_ROOT}/vllm-wheels.tar.gz.sha256"); then
    echo "Hugging Face wheelhouse lookup failed before returning an HTTP status" >&2
    exit 1
  fi
  if [ "$HF_DIGEST_STATUS" = "200" ]; then
    PREBUILT_WHEEL_ARTIFACT_SHA256=$(cat "$HF_DIGEST_FILE")
    [[ "$PREBUILT_WHEEL_ARTIFACT_SHA256" =~ ^[[:xdigit:]]{64}$ ]] || {
      echo "Hugging Face wheel archive digest is malformed" >&2
      exit 1
    }
    curl -fsSL "${HF_WHEEL_ROOT}/vllm-wheels.tar.gz" -o /tmp/hf-vllm-wheels.tar.gz
    PREBUILT_WHEEL_ARTIFACT_URI=file:///tmp/hf-vllm-wheels.tar.gz
    WHEEL_SOURCE=prebuilt-wheelhouse
    echo "using content-addressed Hugging Face wheelhouse ${HF_WHEEL_REPOSITORY}/${EXPECTED_WHEEL_MANIFEST_SHA256}"
  elif [ "$HF_DIGEST_STATUS" = "404" ]; then
    WHEEL_SOURCE=wheel-builder
    echo "no matching Hugging Face wheelhouse; compiling native wheels"
  else
    echo "Hugging Face wheelhouse lookup returned HTTP ${HF_DIGEST_STATUS}" >&2
    exit 1
  fi
fi

if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  : "${PREBUILT_WHEEL_ARTIFACT_URI:?}"
  : "${PREBUILT_WHEEL_ARTIFACT_SHA256:?}"
  [[ "$PREBUILT_WHEEL_ARTIFACT_URI" == s3://* || "$PREBUILT_WHEEL_ARTIFACT_URI" == https://* || "$PREBUILT_WHEEL_ARTIFACT_URI" == file://* ]] || {
    echo "PREBUILT_WHEEL_ARTIFACT_URI must use s3://, https://, or file://" >&2
    exit 2
  }
  [[ "$PREBUILT_WHEEL_ARTIFACT_SHA256" =~ ^[[:xdigit:]]{64}$ ]] || {
    echo "PREBUILT_WHEEL_ARTIFACT_SHA256 must be a SHA-256 digest" >&2
    exit 2
  }
elif [ "$WHEEL_SOURCE" != "wheel-builder" ]; then
  echo "unsupported WHEEL_SOURCE: $WHEEL_SOURCE" >&2
  exit 2
fi

NATIVE_ARCHIVE_SHA256=not-applicable
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  case "$PREBUILT_WHEEL_ARTIFACT_URI" in
    s3://*)
      python3 -m pip install --no-cache-dir fsspec==2026.4.0 s3fs==2026.4.0
      python3 - "$PREBUILT_WHEEL_ARTIFACT_URI" /tmp/vllm-wheels.tar.gz <<'PY'
import shutil
import sys

import fsspec

with fsspec.open(sys.argv[1], "rb") as source, open(sys.argv[2], "wb") as output:
    shutil.copyfileobj(source, output)
PY
      ;;
    https://*) curl -fsSL "$PREBUILT_WHEEL_ARTIFACT_URI" -o /tmp/vllm-wheels.tar.gz ;;
    file://*) install -m 0644 "${PREBUILT_WHEEL_ARTIFACT_URI#file://}" /tmp/vllm-wheels.tar.gz ;;
  esac
  NATIVE_ARCHIVE_SHA256=$(sha256sum /tmp/vllm-wheels.tar.gz | cut -d ' ' -f 1)
  if [ "${NATIVE_ARCHIVE_SHA256,,}" != "${PREBUILT_WHEEL_ARTIFACT_SHA256,,}" ]; then
    echo "wheel artifact SHA-256 mismatch" >&2
    exit 1
  fi

  ARTIFACT_DIR=$(mktemp -d /tmp/vllm-wheel-artifact.XXXXXX)
  tar -xzf /tmp/vllm-wheels.tar.gz -C "$ARTIFACT_DIR"
  ARTIFACT_WHEELS="${ARTIFACT_DIR}/wheels"

  verify_wheelhouse "$ARTIFACT_WHEELS"

  mkdir -p "$WHEELHOUSE"
  find "$WHEELHOUSE" -maxdepth 1 -type f \
    \( -name '*.whl' -o -name MANIFEST -o -name SHA256SUMS \) -delete
  install -m 0644 "$ARTIFACT_WHEELS"/MANIFEST \
    "$ARTIFACT_WHEELS"/SHA256SUMS \
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
crane export --platform "$KANIKO_PLATFORM" gcr.io/kaniko-project/executor:latest kaniko-rootfs.tar
# Extract only Kaniko. Expanding the complete image over the Iris task root
# touches read-only pseudo-filesystems such as /sys and masks real tar errors.
tar -xf kaniko-rootfs.tar -C / kaniko
test -x /kaniko/executor

export DOCKER_CONFIG=/kaniko/.docker
REGISTRY_USER="$REGISTRY_USER" REGISTRY_TOKEN="$REGISTRY_TOKEN" \
  bash "${SCRIPT_DIR}/write_registry_auth.sh" "$REGISTRY_HOST" "$DOCKER_CONFIG"
unset REGISTRY_TOKEN

# When we pay the nvcc compile, keep a minimal wheel-only image before building
# the runtime layers. The same files are archived by manifest digest on Hugging
# Face when requested, so a later build can reuse them even if the registry cache
# is unavailable.
if [ "$WHEEL_SOURCE" = "wheel-builder" ]; then
  set -x
  /kaniko/executor \
    --context "dir://${DOCKER_CONTEXT}" \
    --dockerfile "$DOCKERFILE" \
    --target wheel-artifact \
    --build-arg WHEEL_SOURCE="$WHEEL_SOURCE" \
    --build-arg INSTALL_MEGATRON="$INSTALL_MEGATRON" \
    --build-arg GITSHA="$GITSHA" \
    --build-arg HARBOR_COMMIT="$HARBOR_COMMIT" \
    --build-arg VLLM_NATIVE_DONOR_ARCHIVE_SHA256="$NATIVE_ARCHIVE_SHA256" \
    --skip-unused-stages \
    --compressed-caching=false \
    "${REGISTRY_RETRY_FLAGS[@]}" \
    "${CACHE_FLAGS[@]}" \
    --destination "${IMAGE_REPOSITORY}:wheels-${GITSHA}${ARCH_TAG_SUFFIX}"
  set +x
  WHEEL_IMAGE="${IMAGE_REPOSITORY}:wheels-${GITSHA}${ARCH_TAG_SUFFIX}"
  EXPORTED_WHEELHOUSE=$(mktemp -d /tmp/exported-wheelhouse.XXXXXX)
  crane export --platform "$KANIKO_PLATFORM" "$WHEEL_IMAGE" - | tar -xf - -C "$EXPORTED_WHEELHOUSE" wheels
  verify_wheelhouse "$EXPORTED_WHEELHOUSE/wheels"
  echo "preserved wheel-only image as $WHEEL_IMAGE"

  if [ -n "$HF_WHEEL_REPOSITORY" ]; then
    PUBLISH_DIR=$(mktemp -d /tmp/published-wheelhouse.XXXXXX)
    tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
      -czf "$PUBLISH_DIR/vllm-wheels.tar.gz" -C "$EXPORTED_WHEELHOUSE" wheels
    install -m 0644 "$EXPORTED_WHEELHOUSE/wheels/MANIFEST" "$PUBLISH_DIR/MANIFEST"
    install -m 0644 "$EXPORTED_WHEELHOUSE/wheels/SHA256SUMS" "$PUBLISH_DIR/SHA256SUMS"
    sha256sum "$PUBLISH_DIR/vllm-wheels.tar.gz" | cut -d ' ' -f 1 > "$PUBLISH_DIR/vllm-wheels.tar.gz.sha256"
    HF_PATH="wheelhouses/${WHEEL_PLATFORM_TAG}/${EXPECTED_WHEEL_MANIFEST_SHA256}"
    python3 -m pip install --no-cache-dir huggingface-hub==0.35.3
    HF_HUB_DISABLE_PROGRESS_BARS=1 hf upload "$HF_WHEEL_REPOSITORY" "$PUBLISH_DIR" "$HF_PATH" \
      --repo-type dataset --commit-message "Add ${WHEEL_PLATFORM_TAG} wheelhouse ${EXPECTED_WHEEL_MANIFEST_SHA256}"
    python3 - "$HF_WHEEL_REPOSITORY" "$HF_PATH/vllm-wheels.tar.gz" "$PUBLISH_DIR/vllm-wheels.tar.gz" <<'PY'
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi

remote = HfApi(token=os.environ["HF_TOKEN"]).get_paths_info(sys.argv[1], [sys.argv[2]], repo_type="dataset")
if len(remote) != 1 or remote[0].size != Path(sys.argv[3]).stat().st_size:
    raise RuntimeError("Hugging Face wheel archive size does not match the uploaded artifact")
print(f"verified Hugging Face wheel artifact: {sys.argv[1]}/{sys.argv[2]} ({remote[0].size} bytes)")
PY
  fi
fi

unset HF_TOKEN
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
  "${REGISTRY_RETRY_FLAGS[@]}" \
  "${CACHE_FLAGS[@]}" \
  "${DESTINATIONS[@]}"
