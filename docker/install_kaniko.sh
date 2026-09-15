#!/usr/bin/env bash
# Install architecture-matched crane and unpack the selected Kaniko executor.
set -euo pipefail

CRANE_ASSET_ARCH="${1:?crane release architecture}"
KANIKO_PLATFORM="${2:?kaniko platform}"
CRANE_VERSION=v0.20.2

cd /tmp
curl -fsSL \
  "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_${CRANE_ASSET_ARCH}.tar.gz" \
  -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane
# Kaniko's tag is a multi-architecture manifest, while crane defaults to
# linux/amd64. Always select the build host's platform explicitly.
crane export --platform "$KANIKO_PLATFORM" gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true
test -x /kaniko/executor
