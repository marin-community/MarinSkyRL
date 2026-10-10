#!/usr/bin/env bash
set -euo pipefail

qualification_root="$PWD"
bash cloud/iris/bootstrap_runtime.sh "$qualification_root" /tmp/hero-venv /tmp/hero-runtime.env megatron-export development
source /tmp/hero-runtime.env
export PYTHONPATH="$qualification_root/skyrl-train:$qualification_root/skyrl-gym:$qualification_root"
export SKYRL_HOME="$qualification_root"

# The shipping baseline has known metadata exclusions. Keep their original
# diagnostics; every candidate profile must pass the dependency check.
if [[ "$HERO_STACK_ROLE" == candidate ]]; then
  uv pip check --python /tmp/hero-venv/bin/python
else
  uv pip check --python /tmp/hero-venv/bin/python > /tmp/baseline-pip-check.log 2>&1 || true
  cat /tmp/baseline-pip-check.log
fi

tokenizer_arguments=()
if [[ "$HERO_MODEL_FAMILY" == hero ]]; then
  tokenizer_arguments=(--tokenizer marin-community/marin-tokenizer \
    --tokenizer-revision a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2)
fi
/tmp/hero-venv/bin/python hero_qualification.py --metadata-only \
  --source "$HERO_SOURCE" "${tokenizer_arguments[@]}"

/tmp/hero-venv/bin/python task_runtime_identity.py \
  --expected-commit "$HERO_SOURCE_REVISION" --expected-digest "$HERO_RUNTIME_BUNDLE_SHA256" \
  --source-manifest-sha256 "$HERO_SOURCE_MANIFEST_SHA256"

export HERO_OPTIMIZER=MuonH
export HERO_SKIP_CHECKPOINT=1
exec /tmp/hero-venv/bin/python hero_task_runtime.py \
  --rendezvous-dir "$HERO_OUTPUT/rendezvous" --ray-log-dir "$HERO_OUTPUT/ray-logs" \
  --rendezvous-timeout 7200 --cluster-join-timeout 7200 --driver-liveness-timeout 3600 \
  -- /tmp/hero-venv/bin/python hero_qualification.py --source "$HERO_SOURCE" \
  --model-family "$HERO_MODEL_FAMILY" --output "$HERO_OUTPUT" \
  --nodes "$HERO_POLICY_NODES" --gpus "$HERO_GPUS_PER_NODE" --tp 1 \
  --pp "$HERO_PP" --ep "$HERO_EP" --cp "$HERO_CP" --batch 128 \
  --contexts "$HERO_FIXED_CONTEXT"
