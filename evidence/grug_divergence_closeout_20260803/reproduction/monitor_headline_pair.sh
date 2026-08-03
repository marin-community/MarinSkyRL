#!/usr/bin/env bash
set -u

IRIS_PY=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731/.venv/bin/python
IRIS_CONFIG=/home/romain/dev/marin-wt/grug-training-perf-gap-20260731/lib/iris/config/cw-rno2a.yaml
JOB=/romain/grug-paired-eager-grouped-2dd905e-s1-20260803
WRAPPER='import os; os.chdir("/home/romain/dev/marin-wt/grug-training-perf-gap-20260731"); from rigging.filesystem.cluster_config import store_configs; store_configs(); from iris.cli.main import main; main()'

while true; do
  echo "MONITOR_CHECK_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if output=$("$IRIS_PY" -c "$WRAPPER" --config "$IRIS_CONFIG" job summary "$JOB" 2>&1); then
    echo "$output"
    if ! grep -q '^State: running' <<<"$output"; then
      exit 0
    fi
  else
    echo "$output"
    echo "MONITOR_TRANSIENT_ERROR=1"
  fi
  sleep 55
done
