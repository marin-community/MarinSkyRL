#!/usr/bin/env bash
# Write a minimal Docker credential file for one selected registry host.
set +x
set -euo pipefail

: "${REGISTRY_USER:?}"
: "${REGISTRY_TOKEN:?}"

REGISTRY_HOST="${1:?registry host}"
DOCKER_CONFIG_DIR="${2:?Docker config directory}"
if [[ "$REGISTRY_HOST" == */* ]] || [ -z "$REGISTRY_HOST" ]; then
  echo "registry host must not contain a repository path" >&2
  exit 2
fi

install -d -m 0700 "$DOCKER_CONFIG_DIR"
AUTH=$(printf '%s:%s' "$REGISTRY_USER" "$REGISTRY_TOKEN" | base64 | tr -d '\n')
printf '{"auths":{"%s":{"auth":"%s"}}}\n' "$REGISTRY_HOST" "$AUTH" > "$DOCKER_CONFIG_DIR/config.json"
chmod 0600 "$DOCKER_CONFIG_DIR/config.json"
unset AUTH REGISTRY_TOKEN
