#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:?candidate image digest is required}
PRODUCT_ROOT=/home/romain/dev/marin-wt/grug-moe-execution-20260731
EVIDENCE_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
PRODUCT_SHA=fbb1fc8378601e0346d00d186809f10d1ad0360d
HARNESS_SHA=2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG=${IRIS_CONFIG:-"$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"}
JOB_NAME=${JOB_NAME:-grug-paired-preflight-fbb1fc8-s1-20260803}
RUNNER="$PRODUCT_ROOT/.agents/tmp/run_candidate_preflight_fbb1fc8.sh"

test "$(git -C "$PRODUCT_ROOT" rev-parse HEAD)" = "$PRODUCT_SHA"
git -C "$PRODUCT_ROOT" diff --quiet
git -C "$PRODUCT_ROOT" diff --cached --quiet
git -C "$EVIDENCE_ROOT" cat-file -e "$HARNESS_SHA^{commit}"
test "$(sha256sum "$PRODUCT_ROOT/skyrl-train/skyrl_train/models/grug_moe.py" | cut -d' ' -f1)" = 2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
bash -n "$RUNNER"

SUBMIT_DIR=$(mktemp -d /tmp/grug-correctness-preflight-fbb1fc8.XXXXXX)
trap 'rm -rf -- "$SUBMIT_DIR"' EXIT
git -C "$EVIDENCE_ROOT" archive "$HARNESS_SHA" | tar -x -C "$SUBMIT_DIR"
cp "$PRODUCT_ROOT/skyrl-train/skyrl_train/models/grug_moe.py" "$SUBMIT_DIR/skyrl-train/skyrl_train/models/grug_moe.py"
cp "$RUNNER" "$SUBMIT_DIR/run_candidate_preflight_fbb1fc8.sh"
mkdir "$SUBMIT_DIR/config"
cp "$MARIN_ROOT"/config/*.yaml "$SUBMIT_DIR/config/"

test "$(sha256sum "$SUBMIT_DIR/skyrl-train/scripts/grug_fixed_replay_benchmark.py" | cut -d' ' -f1)" = b46a8d3e2c0516032b8ca9466b047b911f0ec50d1a527df393878c2522049404
test "$(sha256sum "$SUBMIT_DIR/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py" | cut -d' ' -f1)" = c6a954f2cb69996efcfa68fdbac4e43b63955f1f01c5539bf6ed41b1aa7d15b1
test "$(sha256sum "$SUBMIT_DIR/skyrl-train/skyrl_train/models/grug_moe.py" | cut -d' ' -f1)" = 2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
test "$(sha256sum "$SUBMIT_DIR/cloud/iris/start_rl_iris_controller.py" | cut -d' ' -f1)" = f2e8484de7d5566f7e39d6d7e8ef3c03744960a53dd9a39c20a9dfba4db6d2ba

cd "$SUBMIT_DIR"
"$IRIS_BIN" --config "$IRIS_CONFIG" job run \
  --enable-extra-resources \
  --gpu H100x8 \
  --cpu 24 \
  --memory 1200GB \
  --disk 4000GB \
  --priority interactive \
  --max-retries 0 \
  --timeout 86400 \
  --no-wait \
  --job-name "$JOB_NAME" \
  --task-image "$IMAGE" \
  -e IMAGE "$IMAGE" \
  -e HF_HOME /hf/cache \
  -e HF_HUB_OFFLINE 1 \
  -e TRANSFORMERS_OFFLINE 1 \
  -e NCCL_SOCKET_IFNAME '^ibs,ibp,lo,docker,veth,cilium,lxc' \
  -e PYTHONPATH /app:/app/skyrl-train:/opt/skyrl/skyrl-train \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash /app/run_candidate_preflight_fbb1fc8.sh
