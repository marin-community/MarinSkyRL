#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: source resolve_runtime.sh INSTALL_MODE" >&2
  return 2
fi

install_mode="$1"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-env}"
RUNTIME_ENV_FILE="$NIGHTLY_RL_ENV/marinskyrl-runtime.sh"
PYTHON="$NIGHTLY_RL_ENV/bin/python"

echo "::: resolving the frozen MarinSkyRL runtime"
bash "$REPOSITORY_ROOT/cloud/iris/bootstrap_runtime.sh" \
  "$REPOSITORY_ROOT" \
  "$NIGHTLY_RL_ENV" \
  "$RUNTIME_ENV_FILE" \
  fsdp \
  "$install_mode"
source "$RUNTIME_ENV_FILE"
