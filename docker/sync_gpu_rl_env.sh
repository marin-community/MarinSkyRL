#!/usr/bin/env bash

set -euo pipefail

exec uv sync \
  --frozen \
  --no-cache \
  --extra vllm \
  --extra telemetry \
  --no-install-package flash-attn \
  --extra megatron \
  "$@"
