#!/usr/bin/env bash
# The FSDP2 train/eval log-prob parity gate, on H100. Runs INSIDE an Iris GPU task.
#
#   $IRIS --cluster marin job run --target-cluster cw-rno2a \
#     --job-name atqamar-fsdp2-parity --priority batch \
#     --cpu 32 --memory 200GB --disk 400GB --gpu H100x2 --enable-extra-resources \
#     --no-sync --timeout 3600 -- bash -c 'bash skyrl-train/ci/run_fsdp2_train_eval_parity.sh'
#
# 🚨 The grouped_mm arms are EXPECTED TO FAIL while F25 is open -- that is the result we are here
# for. So the pytest exit code is captured and reported rather than allowed to kill the script,
# and every arm's numbers are printed.
set -uo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-parity-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$ENV_DIR" production
cd "$REPOSITORY_ROOT/skyrl-train"
# Ray's zip runtime flattens the repository symlink, so materialise the sibling package.
rm -rf skyrl-gym && cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ⚠️ The frozen TRAINING environment ships no pytest, and a missing pytest exits fast in a way that
# is indistinguishable from a pass from the outside. Install it INTO that env -- not a fresh one --
# so the test imports the same torch and skyrl_train the trainer runs.
"$PYTHON" -m pip install --quiet pytest
"$PYTHON" -c "import pytest, torch, ray; print(f'pytest {pytest.__version__} | torch {torch.__version__} | ray {ray.__version__}')"

# ⚠️ No -x. There are four independent arms and -x would report the first failure while silently
# running none of the others -- the whole 2x2 is the result.
"$PYTHON" -m pytest tests/gpu/test_grug_fsdp2_train_eval_parity.py -q -rA -s -p no:cacheprovider
echo "::: FSDP2 PARITY EXIT=$?"
