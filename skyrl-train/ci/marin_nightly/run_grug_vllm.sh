#!/usr/bin/env bash
# Validate the locked Marin vLLM wheel through rollout, FSDP2 update, weight broadcast, and another rollout.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$NIGHTLY_RL_ENV" development

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "::: running a tiny Grug rollout, FSDP2 update, weight broadcast, and second rollout"
cd "$REPOSITORY_ROOT"
JUNIT_XML="$REPOSITORY_ROOT/grug-vllm-junit.xml"
"$PYTHON" "$REPOSITORY_ROOT/cloud/iris/env_vars.py" \
  run-grug-gpu-gate "$REPOSITORY_ROOT" -- \
  "$PYTHON" -m pytest \
  --junitxml="$JUNIT_XML" \
  'skyrl-train/tests/gpu/test_grug_fsdp2_rl_cycle.py::test_grug_four_gpu_disaggregated_rollout_train_broadcast_rollout[eager]'
"$PYTHON" -c "import xml.etree.ElementTree as ET; cases = ET.parse('$JUNIT_XML').getroot().findall('.//testcase'); assert len(cases) == 1 and cases[0].find('skipped') is None, 'Grug generation gate did not execute'"
