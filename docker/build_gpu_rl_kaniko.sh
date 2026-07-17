#!/usr/bin/env bash
# build_gpu_rl_kaniko.sh — in-cluster kaniko build of the MarinSkyRL gpu-rl image.
#
# Runs INSIDE an iris job whose task-image is docker.io/library/ubuntu:22.04
# (kaniko's executor image is distroless / has no bash, so it cannot be the task
# image directly). We crane-export the kaniko executor rootfs over / and run
# /kaniko/executor. Context = the iris-synced /app bundle (this repo).
# See .claude/skills/build-gpu-rl-image-iris/SKILL.md (in OpenThoughts-Agent).
#
# Required env (passed by the iris launch as -e):
#   DOCKER_USER_ID  ghcr user (penfever)
#   DOCKER_TOKEN    a GitHub PAT with write:packages (NOT the Docker Hub dckr_pat_)
#   GITSHA          MarinSkyRL commit sha for the immutable pinned tag
# Optional:
#   WHEEL_SOURCE       prebuilt-wheelhouse (default) | wheel-builder
#   INSTALL_MEGATRON   0 (default) | 1  -> builds the megatron variant
#   TAG_PREFIX         gpu-rl (default) | gpu-rl-megatron  -> the pinned tag prefix
#   SINGLE_SNAPSHOT    0 (default here) | 1
#   PUSH_FLOATING      0 (default here — experimental; leave :gpu-rl untouched) | 1
#   KANIKO_CACHE       1 (default) | 0
# SECURITY: NO `set -x` before the ghcr token is consumed (would echo DOCKER_TOKEN
# / the base64 AUTH into the R2-persisted finelog). Tracing is enabled AFTER the
# config.json write, so build steps are traced but the secret never is.
set -euo pipefail

: "${DOCKER_USER_ID:?}"
: "${DOCKER_TOKEN:?}"
: "${GITSHA:?}"

WHEEL_SOURCE="${WHEEL_SOURCE:-prebuilt-wheelhouse}"
INSTALL_MEGATRON="${INSTALL_MEGATRON:-0}"
TAG_PREFIX="${TAG_PREFIX:-gpu-rl}"

# SINGLE_SNAPSHOT=0 (default) => per-instruction layers (each small enough to pull
# + retry over the CoreWeave->ghcr egress). =1 collapses to ONE ~16 GB layer that
# CANNOT be pulled (containerd EOFs the single-blob GET) — the build looks green
# but every pod ImagePullBackOffs. --compressed-caching=false keeps multi-layer
# snapshotting within the memory budget.
SINGLE_SNAPSHOT="${SINGLE_SNAPSHOT:-0}"
if [ "$SINGLE_SNAPSHOT" = "1" ]; then SNAPSHOT_FLAG="--single-snapshot"; else SNAPSHOT_FLAG=""; fi

CACHE_REPO=ghcr.io/open-thoughts/openthoughts-agent/cache
if [ "${KANIKO_CACHE:-1}" = "0" ]; then CACHE_FLAGS="--cache=false"; else CACHE_FLAGS="--cache=true --cache-repo=${CACHE_REPO}"; fi

# Consumers pin the DIGEST; the floating :gpu-rl tag is only moved when PUSH_FLOATING=1.
DEST_FLOATING=ghcr.io/open-thoughts/openthoughts-agent:gpu-rl
DEST_PINNED=ghcr.io/open-thoughts/openthoughts-agent:${TAG_PREFIX}-${GITSHA}
FLOATING_DEST_FLAG="--destination ${DEST_FLOATING}"
if [ "${PUSH_FLOATING:-0}" != "1" ]; then FLOATING_DEST_FLAG=""; fi

# --- 1. fetch crane (static binary) ---
apt-get update -y && apt-get install -y --no-install-recommends ca-certificates curl tar
cd /tmp
CRANE_VER=v0.20.2
curl -fsSL "https://github.com/google/go-containerregistry/releases/download/${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz" -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane

# --- 2. crane-export the kaniko executor rootfs over / ---
crane export gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true

# --- 3. write the ghcr auth config AFTER the overlay (kaniko clobbers /kaniko otherwise) ---
export DOCKER_CONFIG=/kaniko/.docker
mkdir -p "$DOCKER_CONFIG"
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$DOCKER_TOKEN" | base64 | tr -d '\n')
cat > "$DOCKER_CONFIG/config.json" <<EOF
{"auths":{"ghcr.io":{"auth":"${AUTH}"}}}
EOF
unset AUTH
set -x  # ghcr PAT consumed — safe to trace the build steps (token never traced)

# --- 3.5. populate docker/wheelhouse/ for the prebuilt-wheelhouse (no-nvcc) path ---
# The iris /app bundle has a 25 MB cap, so the ~900 MB prebuilt wheels cannot ride
# it; fetch them from the public HF mirror into the build context so kaniko's
# `COPY docker/wheelhouse/` (rl stage) finds them => ZERO nvcc. Fail fast if a wheel
# is missing (never silently fall through to a compile).
if [ "$WHEEL_SOURCE" = "prebuilt-wheelhouse" ]; then
  WH=/app/docker/wheelhouse
  mkdir -p "$WH"
  HF_BASE="https://huggingface.co/datasets/laion/gpu-rl-build-wheels/resolve/main"
  FLASH_WHL=flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
  VLLM_WHL=vllm-0.1.dev16611+g76259c63a.d20260625.cu128-cp312-cp312-linux_x86_64.whl
  for f in "$FLASH_WHL" "$VLLM_WHL" MANIFEST; do
    if [ ! -s "$WH/$f" ]; then
      echo "fetching wheelhouse artifact: $f"
      curl -fSL --retry 5 --retry-delay 5 "$HF_BASE/$(printf '%s' "$f" | sed 's/+/%2B/g')" -o "$WH/$f"
    fi
  done
  echo "=== wheelhouse contents ==="; ls -la "$WH"
  test -s "$WH/$FLASH_WHL" && test -s "$WH/$VLLM_WHL" || { echo "FATAL: wheelhouse not populated"; exit 1; }
fi

# --- 4. run kaniko ---
# --skip-unused-stages prunes the wheel-builder (nvcc) stage on the prebuilt path.
exec /kaniko/executor \
  --context dir:///app \
  --dockerfile "${DOCKERFILE:-docker/Dockerfile.gpu-rl}" \
  --build-arg WHEEL_SOURCE="$WHEEL_SOURCE" \
  --build-arg INSTALL_MEGATRON="$INSTALL_MEGATRON" \
  --skip-unused-stages \
  $SNAPSHOT_FLAG \
  --compressed-caching=false \
  $CACHE_FLAGS \
  $FLOATING_DEST_FLAG \
  --destination "${DEST_PINNED}"
