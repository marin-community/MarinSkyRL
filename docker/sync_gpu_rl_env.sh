#!/usr/bin/env bash

set -euo pipefail

policy_extra=(--extra fsdp)
if [[ "${INSTALL_MEGATRON}" == "1" ]]; then
  policy_extra=(--extra megatron)
fi

exec uv sync \
  --frozen \
  --no-cache \
  --extra vllm \
  --extra telemetry \
  --no-install-package flash-attn \
  "${policy_extra[@]}" \
  "$@"
