#!/usr/bin/env bash
# Submit the learner-benchmark smoke (smoke_task.sh) to LB_NODES 8xH100 nodes through the federated Iris hub.
# Usage: submit_smoke.sh JOB_NAME MODEL_URI OUTPUT_URI [ENSEMBLE_ROOT]
# Run from the repository root with a clean tree; the workspace bundle is the code the task runs.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: submit_smoke.sh JOB_NAME MODEL_URI OUTPUT_URI [ENSEMBLE_ROOT]" >&2
  exit 2
fi
job_name="$1"
model="$2"
output="$3"
ensemble_root="${4:-}"

git diff --quiet HEAD -- || { echo "commit local changes first" >&2; exit 1; }
revision="$(git rev-parse HEAD)"
task_image="$(python3 -c 'import ast, sys; tree = ast.parse(open(sys.argv[1]).read()); print(next(node.value.value for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "IRIS_TASK_IMAGE"))' cloud/iris/runtime_environment.py)"

iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster "${LB_TARGET_CLUSTER:-cw-us-east-02a}" --priority interactive \
  --gpu H100x8 --replicas "${LB_NODES:-1}" --cpu 120 --memory 1500GB --disk "${LB_DISK:-600GB}" --timeout "${LB_TIMEOUT:-10800}" \
  --no-sync --task-image "$task_image" --job-name "$job_name" \
  -e LB_MODEL "$model" -e LB_OUTPUT "$output" -e LB_ENSEMBLE_ROOT "$ensemble_root" \
  -e LB_ENSEMBLE_VERSION "${LB_ENSEMBLE_VERSION:-smoke-gsm8k}" -e LB_ROWS "${LB_ROWS:-128}" \
  -e LB_RUN_NAME "$job_name" -e LB_REVISION "$revision" -e LB_NODES "${LB_NODES:-1}" -e LB_TP "${LB_TP:-1}" -e LB_PP "${LB_PP:-1}" -e LB_EP "${LB_EP:-8}" -e LB_MICRO "${LB_MICRO:-8}" \
  -e LB_WARMUP "${LB_WARMUP:-2}" -e LB_MEASURED "${LB_MEASURED:-5}" -e LB_REFERENCE_CHECK "${LB_REFERENCE_CHECK:-1}" \
  -e LB_LR "${LB_LR:-1e-6}" -e LB_DRIVER_SCRIPT "${LB_DRIVER_SCRIPT:-smoke_driver.sh}" -e LB_ROUTES_CHECKPOINT "${LB_ROUTES_CHECKPOINT:-none}" \
  -e LB_SIZES "${LB_SIZES:-}" \
  -e WANDB_MODE disabled \
  -- bash cloud/iris/learner_bench/smoke_task.sh
