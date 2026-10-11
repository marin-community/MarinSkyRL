#!/usr/bin/env bash
# DPO equivalence experiments for issue 969: distributed backward pass, optimizer update,
# and training-to-vLLM transfer, on the frozen Megatron runtime.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DPO_EQUIVALENCE_ENV="${DPO_EQUIVALENCE_ENV:-$REPOSITORY_ROOT/.iris-dpo-equivalence-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$DPO_EQUIVALENCE_ENV" development megatron

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "::: running the DPO equivalence experiments"
cd "$REPOSITORY_ROOT"
JUNIT_XML="$REPOSITORY_ROOT/dpo-equivalence-junit.xml"
test_targets=(
  skyrl-train/tests/gpu/test_dpo_backward_equivalence.py
  skyrl-train/tests/gpu/test_megatron_optimizer_update_equivalence.py
  skyrl-train/tests/gpu/test_dpo_vllm_transfer_equivalence.py
)
if [[ -n "${DPO_EQUIVALENCE_TESTS:-}" ]]; then
  test_targets=("$DPO_EQUIVALENCE_TESTS")
fi
"$PYTHON" "$REPOSITORY_ROOT/marinskyrl/environment_contract.py" \
  run-grug-gpu-gate "$REPOSITORY_ROOT" -- \
  "$PYTHON" -m pytest ${DPO_EQUIVALENCE_PYTEST_ARGS:--x} -s \
  --junitxml="$JUNIT_XML" \
  "${test_targets[@]}"
"$PYTHON" -c "import xml.etree.ElementTree as ET; cases = ET.parse('$JUNIT_XML').getroot().findall('.//testcase'); assert cases and all(case.find('skipped') is None for case in cases), 'a DPO equivalence experiment did not execute'"
