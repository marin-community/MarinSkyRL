#!/usr/bin/env bash

# Large cached layers traverse the CoreWeave-to-registry path for several
# minutes. Kaniko defaults registry operations to zero retries.
KANIKO_REGISTRY_RETRY_FLAGS=(
  --image-fs-extract-retry=3
  --image-download-retry=3
  --push-retry=3
)

install_kaniko_build_packages() {
  apt-get update -y
  apt-get install -y --no-install-recommends "$@"
}

prepare_kaniko_executor_and_registry() {
  local crane_asset_arch="${1:?crane asset architecture}"
  local kaniko_platform="${2:?kaniko platform}"
  local registry_host="${3:?registry host}"
  local helper_dir
  helper_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

  bash "${helper_dir}/bootstrap_kaniko.sh" "$crane_asset_arch" "$kaniko_platform"
  export DOCKER_CONFIG=/kaniko/.docker
  REGISTRY_USER="$REGISTRY_USER" REGISTRY_TOKEN="$REGISTRY_TOKEN" \
    bash "${helper_dir}/write_registry_auth.sh" "$registry_host" "$DOCKER_CONFIG"
  unset REGISTRY_TOKEN
}
