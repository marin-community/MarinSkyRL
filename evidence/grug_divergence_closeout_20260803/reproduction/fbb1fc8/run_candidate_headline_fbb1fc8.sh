#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:?candidate image digest is required}
SOURCE_REVISION=fbb1fc8378601e0346d00d186809f10d1ad0360d
MODEL=marin-community/grug-67b-a2b-sft-s2-thinking-step630
MODEL_REVISION=a822321c2c21af099189e7116104b3cf5142c119
MANIFEST=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260731/replay-step-1-global/e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d/manifest.json
MANIFEST_SHA=5d2479bbbdcd4ca04a9f7d11de82ce42830fbae878d734cdc3c4a4f123f93b74
LOGICAL_SHA=e81f387763177ae55faccf9a2747c2568d59c6efcee7f10d752958771e95f50d
RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-fbb1fc8/headline-paired-s1.json
RENDEZVOUS=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/rendezvous/divergence-closeout-fbb1fc8-headline-paired-s1
PY=/opt/openthoughts/envs/rl/bin/python
DRIVER=/app/skyrl-train/scripts/grug_fixed_replay_benchmark.py
WORKER=/app/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py
MOE=/app/skyrl-train/skyrl_train/models/grug_moe.py
BAKED_MOE=/opt/skyrl/skyrl-train/skyrl_train/models/grug_moe.py
CONTROLLER=/app/cloud/iris/start_rl_iris_controller.py

export HF_HOME=/hf/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_SOCKET_IFNAME='^ibs,ibp,lo,docker,veth,cilium,lxc'
export PYTHONPATH=/app:/app/skyrl-train:/opt/skyrl/skyrl-train
export PYTHONUNBUFFERED=1

run_pair() {
  exec "$PY" "$DRIVER" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --source-revision "$SOURCE_REVISION" \
    --image "$IMAGE" \
    --manifest-s3-uri "$MANIFEST" \
    --manifest-sha256 "$MANIFEST_SHA" \
    --logical-batch-sha256 "$LOGICAL_SHA" \
    --mode headline \
    --attention-backend flash_attention_2 \
    --expert-implementation eager \
    --objective paired_matched_ce \
    --result-s3-uri "$RESULT" \
    --sample 1
}

if [[ "${1:-}" == pair ]]; then
  run_pair
fi

test "$(sha256sum "$DRIVER" | cut -d' ' -f1)" = b46a8d3e2c0516032b8ca9466b047b911f0ec50d1a527df393878c2522049404
test "$(sha256sum "$WORKER" | cut -d' ' -f1)" = c6a954f2cb69996efcfa68fdbac4e43b63955f1f01c5539bf6ed41b1aa7d15b1
test "$(sha256sum "$MOE" | cut -d' ' -f1)" = 2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
test "$(sha256sum "$BAKED_MOE" | cut -d' ' -f1)" = 2dd51d842afe6f57fe00337c21945923da68664d9c23678633c578c27504ba93
cmp -s "$MOE" "$BAKED_MOE"
test "$(sha256sum "$CONTROLLER" | cut -d' ' -f1)" = f2e8484de7d5566f7e39d6d7e8ef3c03744960a53dd9a39c20a9dfba4db6d2ba
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
free -h
for file in memory.current memory.max memory.swap.current memory.swap.max; do
  printf '%s=' "$file"
  cat "/sys/fs/cgroup/$file" 2>/dev/null || printf 'unavailable\n'
done

exec "$PY" "$CONTROLLER" \
  --rendezvous-dir "$RENDEZVOUS" \
  --prestage-model "$MODEL" \
  --model-warm-source s3://marin-us-east-02a/models/marin-community--grug-67b-a2b-sft-s2-thinking-step630 \
  -- \
  /bin/bash "$0" pair
