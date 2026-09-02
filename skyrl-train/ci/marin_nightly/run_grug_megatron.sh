#!/usr/bin/env bash
# Validate Grug on the Megatron trainer: HF parity at PP1/PP2/PP2+EP2, a PP2 update with export,
# and a disaggregated PP2 rollout, update, weight broadcast, and second rollout with Marin vLLM.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$NIGHTLY_RL_ENV" development megatron

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "::: running the Grug Megatron parity, training, and serving gates"
cd "$REPOSITORY_ROOT"
JUNIT_XML="$REPOSITORY_ROOT/grug-megatron-junit.xml"
"$PYTHON" "$REPOSITORY_ROOT/cloud/iris/env_vars.py" \
  run-grug-gpu-gate "$REPOSITORY_ROOT" -- \
  "$PYTHON" -m pytest -x -s \
  --junitxml="$JUNIT_XML" \
  "${GRUG_MEGATRON_TESTS:-skyrl-train/tests/gpu/test_grug_megatron.py}"
"$PYTHON" -c "import xml.etree.ElementTree as ET; cases = ET.parse('$JUNIT_XML').getroot().findall('.//testcase'); assert cases and all(case.find('skipped') is None for case in cases), 'a Grug Megatron gate did not execute'"
