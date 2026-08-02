#!/usr/bin/env bash
set -euo pipefail

MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
MSRL_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
EXPECTED_SOURCE=d81f6364ab66947cdf520a3a42a274b586e830da
IMAGE='ghcr.io/marin-community/marinskyrl@sha256:188eb430485f12182f483a7ee1c2c50191898b5a91e0fa6fea9ef183c4b947a6'
IRIS_PY="$MARIN_ROOT/.venv/bin/python"
IRIS_CONFIG="$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"
READBACK_FILE=/tmp/grug-closeout/run_preflight_failure_readback_d81f636.sh
IRIS_WRAPPER='from rigging.filesystem.cluster_config import store_configs; store_configs(); import os; os.chdir("/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801"); from iris.cli.main import main; main()'

test "$(git -C "$MSRL_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE"
test -z "$(git -C "$MSRL_ROOT" status --porcelain)"
bash -n "$READBACK_FILE"
READBACK_B64=$(base64 < "$READBACK_FILE" | tr -d '\n')

cd "$MARIN_ROOT"
"$IRIS_PY" -c "$IRIS_WRAPPER" --config "$IRIS_CONFIG" job run \
  --enable-extra-resources \
  --cpu 8 \
  --memory 32GB \
  --disk 100GB \
  --priority interactive \
  --max-retries 0 \
  --timeout 1800 \
  --no-sync \
  --job-name grug-preflight-failure-readback-d81f636-r2-20260802 \
  --task-image "$IMAGE" \
  -e READBACK_B64 "$READBACK_B64" \
  -e PYTHONPATH /app:/app/skyrl-train:/opt/skyrl/skyrl-train \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash -c 'set -euo pipefail; echo "$READBACK_B64" | base64 -d > /tmp/grug-preflight-failure-readback.sh; exec /bin/bash /tmp/grug-preflight-failure-readback.sh'
