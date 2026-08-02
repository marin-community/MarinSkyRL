#!/usr/bin/env bash
set -euo pipefail

IMAGE='ghcr.io/marin-community/marinskyrl@sha256:188eb430485f12182f483a7ee1c2c50191898b5a91e0fa6fea9ef183c4b947a6'
SOURCE_REVISION=d81f6364ab66947cdf520a3a42a274b586e830da
MODEL=marin-community/grug-67b-a2b-sft-s2-thinking-step630
MODEL_REVISION=a822321c2c21af099189e7116104b3cf5142c119
MANIFEST=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/manifest.json
MANIFEST_SHA=5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74
LOGICAL_SHA=e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d
RESULT_ROOT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260802/paired-d81f636/preflight
RENDEZVOUS=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260802/rendezvous/paired-d81f636-preflight
PY=/opt/openthoughts/envs/rl/bin/python
DRIVER=/app/skyrl-train/scripts/grug_fixed_replay_benchmark.py
VERIFIER=/app/skyrl-train/scripts/verify_grug_eager_grouped_pair.py
VERDICT=/tmp/grug-paired-preflight-verdict.json

export HF_HOME=/hf/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME='^ibs,ibp,lo,docker,veth,cilium,lxc'
export PYTHONPATH=/app:/app/skyrl-train:/opt/skyrl/skyrl-train
export PYTHONUNBUFFERED=1

run_arm() {
  local implementation=$1
  local result_name=$2
  local attribution=$3
  local attribution_args=()
  if [[ "$attribution" == yes ]]; then
    attribution_args+=(--expert-attribution)
  fi
  "$PY" "$DRIVER" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --source-revision "$SOURCE_REVISION" \
    --image "$IMAGE" \
    --manifest-s3-uri "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" \
    --logical-batch-sha256 "$LOGICAL_SHA" \
    --mode preflight \
    --attention-backend flash_attention_2 \
    --expert-implementation "$implementation" \
    --objective matched_ce \
    --result-s3-uri "$RESULT_ROOT/$result_name" \
    --sample 1 \
    "${attribution_args[@]}"
}

if [[ "${1:-}" == sequence ]]; then
  run_arm eager eager-oracle-s1.json no
  run_arm eager eager-instrumented-s1.json yes
  run_arm grouped grouped-instrumented-s1.json yes
  "$PY" "$VERIFIER" \
    --eager-result "$RESULT_ROOT/eager-instrumented-s1.json" \
    --grouped-result "$RESULT_ROOT/grouped-instrumented-s1.json" \
    --instrumentation-oracle-result "$RESULT_ROOT/eager-oracle-s1.json" \
    --output "$VERDICT"
  sha256sum "$VERDICT"
  "$PY" - "$VERDICT" "$RESULT_ROOT/verdict.json" <<'PY'
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config

source = Path(sys.argv[1])
uri = urlparse(sys.argv[2])
kwargs = {"config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"})}
endpoint = os.environ.get("CW_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
access_key = os.environ.get("CW_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
secret_key = os.environ.get("CW_KEY_SECRET") or os.environ.get("AWS_SECRET_ACCESS_KEY")
if endpoint:
    kwargs["endpoint_url"] = endpoint
if access_key:
    kwargs["aws_access_key_id"] = access_key
if secret_key:
    kwargs["aws_secret_access_key"] = secret_key
boto3.client("s3", **kwargs).put_object(
    Bucket=uri.netloc,
    Key=uri.path.lstrip("/"),
    Body=source.read_bytes(),
    ContentType="application/json",
)
print(f"PREFLIGHT_VERDICT_S3={sys.argv[2]}", flush=True)
PY
  exit 0
fi

test "$(sha256sum "$DRIVER" | cut -d' ' -f1)" = 50cb7024c2f8a45440938648b97d3f6294b01a78c745b17712aaf97bb06c9eea
test "$(sha256sum "$VERIFIER" | cut -d' ' -f1)" = 7e9ec080e28b0d49db42c23d782a1e1ddc09ee0cbbeb7768d69b0c81c5b5c9ae
test "$(sha256sum /app/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py | cut -d' ' -f1)" = 0ea142e4f82c02fb246a08c73f42f8384f1c4d7a38b1f8786a4242ae6071f7df
test "$(sha256sum /app/cloud/iris/start_rl_iris_controller.py | cut -d' ' -f1)" = f2e8484de7d5566f7e39d6d7e8ef3c03744960a53dd9a39c20a9dfba4db6d2ba
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

exec "$PY" /app/cloud/iris/start_rl_iris_controller.py \
  --rendezvous-dir "$RENDEZVOUS" \
  --prestage-model "$MODEL" \
  --model-warm-source s3://marin-us-east-02a/models/marin-community--grug-67b-a2b-sft-s2-thinking-step630 \
  -- \
  /bin/bash "$0" sequence
