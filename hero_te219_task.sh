#!/usr/bin/env bash
set -euo pipefail

qualification_root="$PWD"
bash cloud/iris/bootstrap_runtime.sh "$qualification_root" /tmp/hero-venv /tmp/hero-runtime.env megatron development
source /tmp/hero-runtime.env
uv pip check --python /tmp/hero-venv/bin/python
export PYTHONPATH="$qualification_root/skyrl-train:$qualification_root/skyrl-gym:$qualification_root"
export SKYRL_HOME="$qualification_root"

tokenizer_arguments=()
if [[ "$HERO_MODEL_FAMILY" == hero ]]; then
  tokenizer_arguments=(--tokenizer marin-community/marin-tokenizer \
    --tokenizer-revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2)
fi
/tmp/hero-venv/bin/python hero_qualification.py --metadata-only \
  --source "$HERO_SOURCE" "${tokenizer_arguments[@]}"

/tmp/hero-venv/bin/python - <<'PY'
import hashlib
import importlib.metadata
import os
from pathlib import Path
import vllm
from hero_qualification import s3_client, s3_location

assert importlib.metadata.version('vllm') == os.environ['HERO_VLLM_VERSION']
assert '/tmp/hero-venv/' in vllm.__file__
for prefix, destination in [('HERO_ASYNC_DATA', '/tmp/hero-async-train.jsonl'), ('HERO_EVAL_DATA', '/tmp/hero-async-eval.jsonl')]:
    bucket, key = s3_location(os.environ[prefix + '_URI'])
    data = s3_client().get_object(Bucket=bucket, Key=key)['Body'].read()
    assert hashlib.sha256(data).hexdigest() == os.environ[prefix + '_SHA256']
    Path(destination).write_bytes(data)
    print(destination, len(data), hashlib.sha256(data).hexdigest(), flush=True)
PY

/tmp/hero-venv/bin/python task_runtime_identity.py \
  --expected-commit "$HERO_SOURCE_REVISION" --expected-digest "$HERO_RUNTIME_BUNDLE_SHA256" \
  --source-manifest-sha256 "$HERO_SOURCE_MANIFEST_SHA256"

exec /tmp/hero-venv/bin/python hero_task_runtime.py \
  --rendezvous-dir "$HERO_OUTPUT/rendezvous" --ray-log-dir "$HERO_OUTPUT/ray-logs" \
  --rendezvous-timeout 7200 --cluster-join-timeout 7200 --driver-liveness-timeout 3600 \
  -- /tmp/hero-venv/bin/python hero_async.py --source "$HERO_SOURCE" \
  --data /tmp/hero-async-train.jsonl --eval-data /tmp/hero-async-eval.jsonl \
  --model-family "$HERO_MODEL_FAMILY" --prompt-length "$HERO_PROMPT_LENGTH" \
  --response-length "$HERO_RESPONSE_LENGTH" \
  --output "$HERO_OUTPUT" --name "$HERO_RUN_NAME" --variant "$HERO_VARIANT" \
  --policy-nodes "$HERO_POLICY_NODES" --serving-nodes "$HERO_SERVING_NODES" \
  --pp "$HERO_PP" --ep "$HERO_EP" --cp "$HERO_CP" --batch 16 --epochs 1 \
  --max-steps "$HERO_MAX_STEPS" --checkpoint-interval "$HERO_CHECKPOINT_INTERVAL" \
  --resume-path "$HERO_RESUME_PATH" --eval-interval 2 --generation-workers 4 \
  --calibration-samples-per-prompt "${HERO_CALIBRATION_SAMPLES_PER_PROMPT:-1}" \
  --max-staleness-steps 1 --max-buffered-groups 16 \
  --optimizer MuonH --learning-rate 1e-6 --adam-learning-rate 1e-6 --optimizer-offload-fraction 0 \
  --serving-memory-utilization 0.9 --serving-max-seqs 16 --serving-batched-tokens 512 \
  --weight-sync-transport expert_block --expert-block-verify
