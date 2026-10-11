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
# Every checkpoint, tokenizer, and config the tests read is local; hub HEAD requests
# from concurrent spawned ranks only invite rate-limit backoffs.
export HF_HUB_OFFLINE=1
JUNIT_XML="$REPOSITORY_ROOT/dpo-equivalence-junit.xml"
test_targets=(
  skyrl-train/tests/gpu/test_dpo_backward_equivalence.py
  skyrl-train/tests/gpu/test_megatron_optimizer_update_equivalence.py
  skyrl-train/tests/gpu/test_dpo_vllm_transfer_equivalence.py
)
if [[ -n "${DPO_EQUIVALENCE_TESTS:-}" ]]; then
  # shellcheck disable=SC2206
  test_targets=(${DPO_EQUIVALENCE_TESTS})
fi
set +e
"$PYTHON" "$REPOSITORY_ROOT/marinskyrl/environment_contract.py" \
  run-grug-gpu-gate "$REPOSITORY_ROOT" -- \
  "$PYTHON" -m pytest ${DPO_EQUIVALENCE_PYTEST_ARGS:--x} -s \
  --junitxml="$JUNIT_XML" \
  "${test_targets[@]}"
pytest_status=$?
set -e
if [[ -f "$REPOSITORY_ROOT/dpo-diagnostics.log" ]]; then
  # Printed after pytest's own summary so these lines finish the job stream and survive
  # log-tail truncation on failing runs.
  echo "::: dpo diagnostics"
  tail -n 200 "$REPOSITORY_ROOT/dpo-diagnostics.log"
fi
[[ $pytest_status -eq 0 ]] || exit $pytest_status
"$PYTHON" -c "import xml.etree.ElementTree as ET; cases = ET.parse('$JUNIT_XML').getroot().findall('.//testcase'); assert cases and all(case.find('skipped') is None for case in cases), 'a DPO equivalence experiment did not execute'"
