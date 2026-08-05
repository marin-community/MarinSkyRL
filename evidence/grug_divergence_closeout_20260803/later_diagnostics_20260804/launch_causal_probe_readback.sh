#!/usr/bin/env bash
set -euo pipefail

EVIDENCE_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
IMAGE=ghcr.io/marin-community/marinskyrl@sha256:bf136bb210ded34b07b3583de7272f2c534df195bca556debbb38c0664640770
MODE=${MODE:-preflight}
READER="$EVIDENCE_ROOT/.agents/tmp/verify_causal_probe.py"
READER_SHA=dd7b7c0544714ab4bb3dbc5e0c839a7f20bf46cd56d678a73815b28275a32683
IRIS_BIN="$MARIN_ROOT/.venv/bin/iris"
IRIS_CONFIG=${IRIS_CONFIG:-"$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"}

case "$MODE" in
  preflight)
    RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/causal-probe-preflight-r0-mb0-l2-s1-rno-cpu8-mem768.json
    JOB_NAME=grug-causal-probe-preflight-readback-v3-fbb1fc8-r0-mb0-l2-s1-rno-20260804
    ;;
  headline)
    RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260804/route-residual-fbb1fc8/causal-probe-headline-r22-mb54-l2-s1-rno-cpu8-mem768.json
    JOB_NAME=grug-causal-probe-headline-readback-fbb1fc8-r22-mb54-l2-s1-rno-20260804
    ;;
  *)
    printf 'unsupported MODE=%s\n' "$MODE" >&2
    exit 2
    ;;
esac

python -m py_compile "$READER"
test "$(sha256sum "$READER" | cut -d' ' -f1)" = "$READER_SHA"
READER_B64=$(base64 < "$READER" | tr -d '\n')

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY_RUN reader verified: mode=%s result=%s reader_sha=%s cpu=1 config=%s\n' \
    "$MODE" "$RESULT" "$READER_SHA" "$IRIS_CONFIG"
  exit 0
fi

cd "$MARIN_ROOT"
# shellcheck disable=SC2016  # expanded inside the task container
"$IRIS_BIN" --config "$IRIS_CONFIG" job run \
  --cpu 1 \
  --memory 32GB \
  --disk 100GB \
  --enable-extra-resources \
  --priority interactive \
  --max-retries 0 \
  --timeout 1800 \
  --no-wait \
  --no-sync \
  --job-name "$JOB_NAME" \
  --task-image "$IMAGE" \
  -e READER_B64 "$READER_B64" \
  -e MODE "$MODE" \
  -e RESULT "$RESULT" \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash -c 'set -euo pipefail; echo "$READER_B64" | base64 -d > /tmp/verify-causal-probe.py; exec /opt/openthoughts/envs/rl/bin/python /tmp/verify-causal-probe.py "$MODE" "$RESULT"'
