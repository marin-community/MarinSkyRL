#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770}
PRODUCT_ROOT=/home/romain/dev/marin-wt/grug-moe-execution-20260731
EVIDENCE_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
PRODUCT_SHA=fbb1fc8378601e0346d00d186809f10d1ad0360d
HARNESS_BASE=7c3bac451a69d34fa8b8f027ceb91998a6e0ff2c
DRIVER_SHA=4f0d7a468558849a5567d96e1c05d49fcc67d55df085ce82773265cab6481373
WORKER_SHA=cced801b807d5d75ca15b7fc1e81c83121e83ecddba2cbe0807b663fb1cce0eb
MODEL_SHA=2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
RUNNER_SHA=284b9b75011b3e9259c2f94a3d1757b82e8d5aaf54326a9b70704daeb5566ae6
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG=${IRIS_CONFIG:-"$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"}
MODE=${MODE:-preflight}
CPU=${CPU:-8}
MEMORY=${MEMORY:-768GB}
RUNNER="$EVIDENCE_ROOT/.agents/tmp/run_causal_probe_fbb1fc8.sh"

case "$MODE" in
  preflight)
    JOB_NAME=${JOB_NAME:-grug-causal-probe-preflight-fbb1fc8-r0-mb0-l2-s1-rno-cpu8-mem768-20260804}
    RESULT=${RESULT:-s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/causal-probe-preflight-r0-mb0-l2-s1-rno-cpu8-mem768.json}
    RENDEZVOUS=${RENDEZVOUS:-s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/rendezvous/route-residual-fbb1fc8-causal-probe-preflight-r0-mb0-l2-s1-rno-cpu8-mem768}
    replica_args=()
    ;;
  headline)
    JOB_NAME=${JOB_NAME:-grug-causal-probe-headline-fbb1fc8-r22-mb54-l2-s1-rno-cpu8-mem768-20260804}
    RESULT=${RESULT:-s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/causal-probe-headline-r22-mb54-l2-s1-rno-cpu8-mem768.json}
    RENDEZVOUS=${RENDEZVOUS:-s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/rendezvous/route-residual-fbb1fc8-causal-probe-headline-r22-mb54-l2-s1-rno-cpu8-mem768}
    replica_args=(--replicas 4)
    ;;
  *)
    printf 'unsupported MODE=%s\n' "$MODE" >&2
    exit 2
    ;;
esac

test "$(git -C "$PRODUCT_ROOT" rev-parse HEAD)" = "$PRODUCT_SHA"
git -C "$PRODUCT_ROOT" diff --quiet
git -C "$PRODUCT_ROOT" diff --cached --quiet
test "$(git -C "$EVIDENCE_ROOT" rev-parse HEAD)" = "$HARNESS_BASE"
bash -n "$RUNNER"
test "$(sha256sum "$RUNNER" | cut -d' ' -f1)" = "$RUNNER_SHA"

SUBMIT_DIR=$(mktemp -d /tmp/grug-causal-probe.XXXXXX)
trap 'rm -rf -- "$SUBMIT_DIR"' EXIT
git -C "$EVIDENCE_ROOT" archive "$HARNESS_BASE" | tar -x -C "$SUBMIT_DIR"
cp "$EVIDENCE_ROOT/skyrl-train/scripts/grug_fixed_replay_benchmark.py" "$SUBMIT_DIR/skyrl-train/scripts/"
cp "$EVIDENCE_ROOT/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py" \
  "$SUBMIT_DIR/skyrl-train/skyrl_train/workers/fsdp/"
cp "$PRODUCT_ROOT/skyrl-train/skyrl_train/models/grug_moe.py" "$SUBMIT_DIR/skyrl-train/skyrl_train/models/"
cp "$RUNNER" "$SUBMIT_DIR/run_causal_probe_fbb1fc8.sh"
mkdir "$SUBMIT_DIR/config"
cp "$MARIN_ROOT"/config/*.yaml "$SUBMIT_DIR/config/"

test "$(sha256sum "$SUBMIT_DIR/skyrl-train/scripts/grug_fixed_replay_benchmark.py" | cut -d' ' -f1)" = "$DRIVER_SHA"
test "$(sha256sum "$SUBMIT_DIR/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py" | cut -d' ' -f1)" = "$WORKER_SHA"
test "$(sha256sum "$SUBMIT_DIR/skyrl-train/skyrl_train/models/grug_moe.py" | cut -d' ' -f1)" = "$MODEL_SHA"
test "$(sha256sum "$SUBMIT_DIR/run_causal_probe_fbb1fc8.sh" | cut -d' ' -f1)" = "$RUNNER_SHA"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY_RUN bundle verified: mode=%s job=%s result=%s config=%s cpu=%s memory=%s replicas=%s\n' \
    "$MODE" "$JOB_NAME" "$RESULT" "$IRIS_CONFIG" "$CPU" "$MEMORY" "${replica_args[*]:-1}"
  exit 0
fi

cd "$SUBMIT_DIR"
"$IRIS_BIN" --config "$IRIS_CONFIG" job run \
  --enable-extra-resources \
  --gpu H100x8 \
  --cpu "$CPU" \
  --memory "$MEMORY" \
  --disk 4000GB \
  "${replica_args[@]}" \
  --priority interactive \
  --max-retries 0 \
  --timeout 86400 \
  --no-wait \
  --job-name "$JOB_NAME" \
  --task-image "$IMAGE" \
  -e IMAGE "$IMAGE" \
  -e MODE "$MODE" \
  -e RESULT "$RESULT" \
  -e RENDEZVOUS "$RENDEZVOUS" \
  -e HF_HOME /hf/cache \
  -e HF_HUB_OFFLINE 1 \
  -e TRANSFORMERS_OFFLINE 1 \
  -e NCCL_SOCKET_IFNAME '^ibs,ibp,lo,docker,veth,cilium,lxc' \
  -e PYTHONPATH /app:/app/skyrl-train:/opt/skyrl/skyrl-train \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash /app/run_causal_probe_fbb1fc8.sh
