#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770}
PRODUCT_ROOT=/home/romain/dev/marin-wt/grug-moe-execution-20260731
EVIDENCE_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
PRODUCT_SHA=fbb1fc8378601e0346d00d186809f10d1ad0360d
HARNESS_BASE=7c3bac451a69d34fa8b8f027ceb91998a6e0ff2c
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG=${IRIS_CONFIG:-"$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"}
CPU=${CPU:-8}
MEMORY=${MEMORY:-768GB}
JOB_NAME=${JOB_NAME:-grug-route-discriminator-headline-fbb1fc8-s1-rno-cpu8-mem768-20260804}
RESULT=${RESULT:-s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/headline-discriminator-s1-rno-cpu8-mem768.json}
RENDEZVOUS=${RENDEZVOUS:-s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/rendezvous/route-residual-fbb1fc8-headline-s1-rno-cpu8-mem768}
RUNNER="$EVIDENCE_ROOT/.agents/tmp/run_route_discriminator_fbb1fc8.sh"
RUNNER_SHA=39db06d8462c805e6bb3c4f658354c602eccd01727ce16b2747eaf2b263a1e5b

test "$(git -C "$PRODUCT_ROOT" rev-parse HEAD)" = "$PRODUCT_SHA"
git -C "$PRODUCT_ROOT" diff --quiet
git -C "$PRODUCT_ROOT" diff --cached --quiet
test "$(git -C "$EVIDENCE_ROOT" rev-parse HEAD)" = "$HARNESS_BASE"
bash -n "$RUNNER"
test "$(sha256sum "$RUNNER" | cut -d' ' -f1)" = "$RUNNER_SHA"

SUBMIT_DIR=$(mktemp -d /tmp/grug-route-discriminator-headline.XXXXXX)
trap 'rm -rf -- "$SUBMIT_DIR"' EXIT
git -C "$EVIDENCE_ROOT" archive "$HARNESS_BASE" | tar -x -C "$SUBMIT_DIR"
cp "$EVIDENCE_ROOT/skyrl-train/scripts/grug_fixed_replay_benchmark.py" "$SUBMIT_DIR/skyrl-train/scripts/"
cp "$EVIDENCE_ROOT/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py" \
  "$SUBMIT_DIR/skyrl-train/skyrl_train/workers/fsdp/"
cp "$PRODUCT_ROOT/skyrl-train/skyrl_train/models/grug_moe.py" "$SUBMIT_DIR/skyrl-train/skyrl_train/models/"
cp "$RUNNER" "$SUBMIT_DIR/run_route_discriminator_fbb1fc8.sh"
mkdir "$SUBMIT_DIR/config"
cp "$MARIN_ROOT"/config/*.yaml "$SUBMIT_DIR/config/"

test "$(sha256sum "$SUBMIT_DIR/skyrl-train/scripts/grug_fixed_replay_benchmark.py" | cut -d' ' -f1)" = bbdc711b3d26b5127a71b3b8e24f7f3dfb5e00ba94e1a51f28f9bf83111dd084
test "$(sha256sum "$SUBMIT_DIR/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py" | cut -d' ' -f1)" = d66c1c3ee148a8aef0007d1d3e17af4ef522381c0107f836d3c0817805fc0de4
test "$(sha256sum "$SUBMIT_DIR/skyrl-train/skyrl_train/models/grug_moe.py" | cut -d' ' -f1)" = 2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
test "$(sha256sum "$SUBMIT_DIR/run_route_discriminator_fbb1fc8.sh" | cut -d' ' -f1)" = "$RUNNER_SHA"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY_RUN bundle verified: job=%s result=%s config=%s cpu=%s memory=%s\n' \
    "$JOB_NAME" "$RESULT" "$IRIS_CONFIG" "$CPU" "$MEMORY"
  exit 0
fi

cd "$SUBMIT_DIR"
"$IRIS_BIN" --config "$IRIS_CONFIG" job run \
  --enable-extra-resources \
  --gpu H100x8 \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --disk 4000GB \
  --replicas 4 \
  --priority interactive \
  --max-retries 0 \
  --timeout 86400 \
  --no-wait \
  --job-name "$JOB_NAME" \
  --task-image "$IMAGE" \
  -e IMAGE "$IMAGE" \
  -e MODE headline \
  -e RESULT "$RESULT" \
  -e RENDEZVOUS "$RENDEZVOUS" \
  -e HF_HOME /hf/cache \
  -e HF_HUB_OFFLINE 1 \
  -e TRANSFORMERS_OFFLINE 1 \
  -e NCCL_SOCKET_IFNAME '^ibs,ibp,lo,docker,veth,cilium,lxc' \
  -e PYTHONPATH /app:/app/skyrl-train:/opt/skyrl/skyrl-train \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash /app/run_route_discriminator_fbb1fc8.sh
