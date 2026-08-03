#!/usr/bin/env bash
set -euo pipefail

MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
MSRL_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
EXPECTED_SOURCE=2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2
IMAGE='ghcr.io/marin-community/marinskyrl@sha256:24c655d33ebb6ef78b9f9a5db4053f838c2e9d6c98e3adef338cdb87e1c072a2'
IRIS_PY="$MARIN_ROOT/.venv/bin/python"
IRIS_CONFIG="$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"
PAIR_FILE=/tmp/grug-closeout/run_headline_pair_2dd905e.sh
IRIS_WRAPPER='import os; os.chdir("/home/romain/dev/marin-wt/grug-training-perf-gap-20260731"); from rigging.filesystem.cluster_config import store_configs; store_configs(); os.chdir("/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801"); from iris.cli.main import main; main()'

cd "$MSRL_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
test -z "$(git status --porcelain)"
test "$IMAGE" != '__IMAGE__'
test "$(sha256sum skyrl-train/scripts/grug_fixed_replay_benchmark.py | cut -d' ' -f1)" = b46a8d3e2c0516032b8ca9466b047b911f0ec50d1a527df393878c2522049404
test "$(sha256sum skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py | cut -d' ' -f1)" = c6a954f2cb69996efcfa68fdbac4e43b63955f1f01c5539bf6ed41b1aa7d15b1
test "$(sha256sum cloud/iris/start_rl_iris_controller.py | cut -d' ' -f1)" = f2e8484de7d5566f7e39d6d7e8ef3c03744960a53dd9a39c20a9dfba4db6d2ba
bash -n "$PAIR_FILE"
PAIR_B64=$(base64 < "$PAIR_FILE" | tr -d '\n')

"$IRIS_PY" -c "$IRIS_WRAPPER" --config "$IRIS_CONFIG" job run \
  --enable-extra-resources \
  --gpu H100x8 \
  --cpu 48 \
  --memory 1600GB \
  --disk 4000GB \
  --replicas 4 \
  --priority production \
  --max-retries 0 \
  --timeout 86400 \
  --no-wait \
  --no-sync \
  --job-name grug-paired-eager-grouped-2dd905e-s1-20260803 \
  --task-image "$IMAGE" \
  -e PAIR_B64 "$PAIR_B64" \
  -e HF_HOME /hf/cache \
  -e HF_HUB_OFFLINE 1 \
  -e TRANSFORMERS_OFFLINE 1 \
  -e NCCL_SOCKET_IFNAME '^ibs,ibp,lo,docker,veth,cilium,lxc' \
  -e PYTHONPATH /app:/app/skyrl-train:/opt/skyrl/skyrl-train \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash -c 'set -euo pipefail; echo "$PAIR_B64" | base64 -d > /tmp/grug-headline-pair.sh; exec /bin/bash /tmp/grug-headline-pair.sh'
