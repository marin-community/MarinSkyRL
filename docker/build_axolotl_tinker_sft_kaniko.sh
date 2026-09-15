#!/usr/bin/env bash
# Build the Marin-owned Axolotl Tinker-SFT control image in an amd64 Iris task.
set +x
set -euo pipefail

: "${GITSHA:?}"
REGISTRY_USER="${REGISTRY_USER:-${DOCKER_USER_ID:-}}"
REGISTRY_TOKEN="${REGISTRY_TOKEN:-${GHCR_TOKEN:-}}"
: "${REGISTRY_USER:?}"
: "${REGISTRY_TOKEN:?}"

if [ "$(uname -m)" != x86_64 ]; then
  echo "The Tinker SFT control image targets the amd64 Iris H100 slice" >&2
  exit 2
fi
if [[ ! "$GITSHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GITSHA must be the full lowercase commit bundled into the image" >&2
  exit 2
fi
if [ -z "${IRIS_TASK_ID:-}" ] || [ ! -f /app/docker/Dockerfile.axolotl-tinker-sft ]; then
  echo "Run this builder inside a disposable Iris task with the checkout at /app" >&2
  exit 2
fi

IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-ghcr.io/marin-community/marinskyrl}"
REGISTRY_HOST="${IMAGE_REPOSITORY%%/*}"
if [ "$REGISTRY_HOST" = "$IMAGE_REPOSITORY" ]; then
  echo "IMAGE_REPOSITORY must include a registry hostname and repository path" >&2
  exit 2
fi

cd /tmp
CRANE_VERSION=v0.20.2
curl -fsSL \
  "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz" \
  -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane
crane export --platform linux/amd64 gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true
test -x /kaniko/executor

DOCKER_CONFIG_DIR=/kaniko/.docker
REGISTRY_USER="$REGISTRY_USER" REGISTRY_TOKEN="$REGISTRY_TOKEN" \
  /app/docker/write_registry_auth.sh "$REGISTRY_HOST" "$DOCKER_CONFIG_DIR"
unset REGISTRY_TOKEN

exec env DOCKER_CONFIG="$DOCKER_CONFIG_DIR" /kaniko/executor \
  --context dir:///app \
  --dockerfile /app/docker/Dockerfile.axolotl-tinker-sft \
  --build-arg GITSHA="$GITSHA" \
  --cache=true \
  --cache-repo="${IMAGE_REPOSITORY}/cache-axolotl-tinker-sft" \
  --image-fs-extract-retry=3 \
  --image-download-retry=3 \
  --push-retry=3 \
  --destination "${IMAGE_REPOSITORY}:axolotl-tinker-sft-${GITSHA}"
