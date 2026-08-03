#!/usr/bin/env bash
set -euo pipefail

MARIN_ROOT=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731
MSRL_ROOT=/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801
EXPECTED_SOURCE=2dd905e29597b848f912dd5cdaf2cebdfbf3d0c2
IMAGE='ghcr.io/marin-community/marinskyrl@sha256:24c655d33ebb6ef78b9f9a5db4053f838c2e9d6c98e3adef338cdb87e1c072a2'
READER="$MSRL_ROOT/evidence/grug_divergence_closeout_20260803/reproduction/verify_headline_artifact.py"
RESULT=s3://marin-us-east-02a/iris/grug-training-perf-gap/20260803/divergence-closeout-2dd905e/headline-paired-s1.json
IRIS_PY="$MARIN_ROOT/.venv/bin/python"
IRIS_CONFIG="$MARIN_ROOT/lib/iris/config/cw-rno2a.yaml"
IRIS_WRAPPER='import os; os.chdir("/home/romain/dev/marin-wt/grug-training-perf-gap-20260731"); from rigging.filesystem.cluster_config import store_configs; store_configs(); os.chdir("/home/romain/dev/marin-wt/grug-training-gap-attribution-msrl-20260801"); from iris.cli.main import main; main()'

cd "$MSRL_ROOT"
git cat-file -e "$EXPECTED_SOURCE^{commit}"
git merge-base --is-ancestor "$EXPECTED_SOURCE" HEAD
test -z "$(git status --porcelain)"
test "$IMAGE" != '__IMAGE__'
python -m py_compile "$READER"
READER_B64=$(base64 < "$READER" | tr -d '\n')

"$IRIS_PY" -c "$IRIS_WRAPPER" --config "$IRIS_CONFIG" job run \
  --cpu 8 \
  --memory 32GB \
  --disk 100GB \
  --enable-extra-resources \
  --priority interactive \
  --max-retries 0 \
  --timeout 1800 \
  --no-wait \
  --no-sync \
  --job-name grug-paired-strict-readback-2dd905e-s1-20260803 \
  --task-image "$IMAGE" \
  -e READER_B64 "$READER_B64" \
  -e RESULT "$RESULT" \
  -e PYTHONUNBUFFERED 1 \
  -- \
  /bin/bash -c 'set -euo pipefail; echo "$READER_B64" | base64 -d > /tmp/read-paired-artifact.py; exec /opt/openthoughts/envs/rl/bin/python /tmp/read-paired-artifact.py "$RESULT" /tmp/headline.raw.json /tmp/headline.summary.json'
