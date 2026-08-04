#!/usr/bin/env bash
# Validate the locked Marin vLLM wheel by loading a real Grug checkpoint and generating tokens.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-env}"
RUNTIME_ENV_FILE="$NIGHTLY_RL_ENV/marinskyrl-runtime.sh"
PYTHON="$NIGHTLY_RL_ENV/bin/python"

echo "::: resolving the frozen MarinSkyRL runtime"
bash "$REPOSITORY_ROOT/cloud/iris/bootstrap_runtime.sh" \
  "$REPOSITORY_ROOT" \
  "$NIGHTLY_RL_ENV" \
  "$RUNTIME_ENV_FILE" \
  fsdp \
  development
source "$RUNTIME_ENV_FILE"
export PYTHONPATH="$REPOSITORY_ROOT/skyrl-gym:$REPOSITORY_ROOT/skyrl-train${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=0

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "::: running a tiny Grug rollout, FSDP2 update, weight broadcast, and second rollout"
cd "$REPOSITORY_ROOT"
JUNIT_XML="$REPOSITORY_ROOT/grug-vllm-junit.xml"
"$PYTHON" -m pytest \
  --junitxml="$JUNIT_XML" \
  'skyrl-train/tests/gpu/test_grug_fsdp2_rl_cycle.py::test_grug_four_gpu_disaggregated_rollout_train_broadcast_rollout[eager]'
"$PYTHON" -c "import xml.etree.ElementTree as ET; cases = ET.parse('$JUNIT_XML').getroot().findall('.//testcase'); assert len(cases) == 1 and cases[0].find('skipped') is None, 'Grug generation gate did not execute'"
