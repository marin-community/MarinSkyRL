#!/usr/bin/env bash
# A12 — the flash-attention correctness gate, on one H100, inside an Iris job.
# Gates `--flash-attn`: the arm reports 1.30x on policy_forward and fwd_logprobs together, and the
# only thing separating a real speedup from a changed forward is a parity check against the eager
# attention path at the same shapes. Builds the same frozen runtime the nightly uses, then runs it.
set -euo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-fagate-env}"
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

echo "::: A12 flash-attention correctness gate"
# `set -e` would abort here on a failing gate, so the status line after it could only ever
# print 0 -- a health signal true by construction, which is the defect this branch exists to
# eliminate. A gate's failure IS its result, so disable the guard across the gate only,
# capture the status, and exit on it.
set +e
"$PYTHON" -m pytest tests/gpu/test_grug_flash_attention.py -v --no-header -p no:cacheprovider
status=$?
set -e
echo "::: FA EXIT=$status"
exit "$status"
