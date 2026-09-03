#!/usr/bin/env bash
# A3 — the grouped_mm parity gate, on one H100, inside an Iris job.
# Gates A2 (`--grouped-mm`): the native-Grug grouped path carries pytorch#186365 (uninitialised
# ALIGN_SIZE_M tail rows) and had NO EP=1 coverage -- tests/gpu/test_grug_fsdp2_rl_cycle.py binds
# use_grouped_mm to expert_model_parallel_size > 1, so the exact combination our arm runs is never
# exercised. Builds the same frozen runtime the nightly uses, then runs the gate.
set -euo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-parity-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$ENV_DIR" production

cd "$REPOSITORY_ROOT/skyrl-train"
# Ray's zip runtime flattens the sibling symlink; materialise it before setting source paths.
rm -rf skyrl-gym && cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "::: GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# The frozen runtime is the training environment; it ships no pytest. Install it INTO that env
# rather than a fresh one, so the test imports the same torch/skyrl_train the trainer runs.
echo "::: installing pytest into the frozen runtime"
uv pip install --python "$PYTHON" --quiet pytest
"$PYTHON" -m pytest --version

echo "::: A3 parity gate"
# `set -e` would abort here on a failing gate, so the status line after it could only ever
# print 0 -- a health signal true by construction, which is the defect this branch exists to
# eliminate. A gate's failure IS its result, so disable the guard across the gate only,
# capture the status, and exit on it.
set +e
"$PYTHON" -m pytest tests/gpu/gpu_ci/test_grug_grouped_mm_parity.py -v --no-header -p no:cacheprovider
status=$?
set -e
echo "::: A3 EXIT=$status"
exit "$status"
