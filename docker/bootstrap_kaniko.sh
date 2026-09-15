#!/usr/bin/env bash
set -euo pipefail

CRANE_ASSET_ARCH="${1:?crane asset architecture}"
KANIKO_PLATFORM="${2:?kaniko platform}"
CRANE_VERSION=v0.20.2

cd /tmp
curl -fsSL \
  "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_${CRANE_ASSET_ARCH}.tar.gz" \
  -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane
crane export --platform "$KANIKO_PLATFORM" gcr.io/kaniko-project/executor:latest kaniko-rootfs.tar
# Extract only Kaniko. Expanding the complete image over an Iris task root
# touches read-only pseudo-filesystems such as /sys and masks real tar errors.
tar -xf kaniko-rootfs.tar -C / kaniko
test -x /kaniko/executor
