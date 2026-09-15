#!/usr/bin/env bash
# Build a Marin-native Open-MOPD experiment image under an isolated tag.
# SHELLOPTS can propagate xtrace, so disable it before reading credentials.
set +x
set -euo pipefail

: "${GITSHA:?}"
: "${REGISTRY_USER:?}"
: "${REGISTRY_TOKEN:?}"

if [ "$(uname -m)" != x86_64 ]; then
  echo "Open-MOPD experiments currently target the x86_64 Iris H100 slice" >&2
  exit 2
fi
if [[ ! "$GITSHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "GITSHA must be the full lowercase commit SHA bundled into the Iris task" >&2
  exit 2
fi

# Reuse the maintained source and frozen root environment. The isolated tag is
# an experiment-selection boundary, not a second dependency closure.
IMAGE_REPOSITORY="${OPEN_MOPD_IMAGE_REPOSITORY:-us-east1-docker.pkg.dev/hai-gcp-models/marin/marinskyrl}"
WHEEL_SOURCE="${WHEEL_SOURCE:-auto}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec env \
  REGISTRY_USER="$REGISTRY_USER" \
  REGISTRY_TOKEN="$REGISTRY_TOKEN" \
  IMAGE_REPOSITORY="$IMAGE_REPOSITORY" \
  TAG_PREFIX=opd-repro \
  DOCKERFILE=docker/Dockerfile.gpu-rl \
  INSTALL_MEGATRON=0 \
  PUBLISH_WHEELHOUSE_HF=1 \
  HF_WHEEL_REPOSITORY=open-athena/marinskyrl-gpu-wheelhouse \
  WHEEL_SOURCE="$WHEEL_SOURCE" \
  bash "${SCRIPT_DIR}/build_gpu_rl_kaniko.sh"
