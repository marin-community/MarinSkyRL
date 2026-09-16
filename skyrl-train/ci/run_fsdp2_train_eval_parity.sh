#!/usr/bin/env bash
# The grouped-MoE combine gates, on two H100s. Runs INSIDE an Iris GPU task.
#
#   $IRIS --cluster marin job run --target-cluster <peer with room> \
#     --job-name <user>-fsdp2-parity --priority batch \
#     --cpu 32 --memory 200GB --disk 400GB --gpu H100x2 --enable-extra-resources \
#     --no-sync --timeout 5400 -- bash -c 'bash skyrl-train/ci/run_fsdp2_train_eval_parity.sh'
#
# Three stages, each reported with its own exit line so a failure in one cannot hide another:
#   1. ci/probe_combine_order.py           -- op-level: does the combine's order matter, and does the
#                                             former scatter_add vary it (one GPU, a minute).
#   2. the one-GPU parity gates            -- the Grug grouped_mm gates and the Qwen grouped-GEMM swap's
#                                             parity arms, both paths sharing the fixed-order combine.
#   3. the FSDP2 train/eval parity gate    -- the real worker path on two GPUs, six arms including the
#                                             26-layer one at the production sequence length.
set -uo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-parity-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$ENV_DIR" production
# resolve_runtime.sh is sourced and sets -e, which would end this script at the first failing stage
# and skip the ones after it (that is how the first run never reached the FSDP2 gate). Every stage
# below must run; the exit status is assembled at the end.
set +e
cd "$REPOSITORY_ROOT/skyrl-train"
# Ray's zip runtime flattens the repository symlink, so materialise the sibling package.
rm -rf skyrl-gym && cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The FSDP2 gate skips every arm when it cannot see two GPUs, and an all-skip pytest exits 0. Refuse
# here so the job cannot read succeeded having measured nothing.
gpu_count="$(nvidia-smi -L | wc -l)"
if [ "$gpu_count" -lt 2 ]; then
  echo "::: NEED 2 GPUs, found $gpu_count"
  exit 1
fi

# The frozen TRAINING environment ships no pytest and no pip; a missing pytest exits fast in a way that
# is indistinguishable from a pass from the outside. Install it INTO that env with uv so the tests
# import the same torch and skyrl_train the trainer runs.
uv pip install --quiet --python "$PYTHON" pytest
"$PYTHON" -c "import pytest, torch, ray; print(f'pytest {pytest.__version__} | torch {torch.__version__} | ray {ray.__version__}')"

"$PYTHON" ci/probe_combine_order.py
probe_status=$?
echo "::: COMBINE PROBE EXIT=$probe_status"

# No -x anywhere below: every arm is an independent result and -x would report the first failure while
# silently running none of the others. G3b-2 stays out: it asserts the router-replay controller is gone
# after a grad-enabled forward, which the Stage-7 teardown deferral (model_wrapper.py) made false, and it
# fails on the branch this one started from rather than because of it. G3b-5 is back in -- it was dropped
# for inheriting G3b-2's leaked controller, but this filter already excludes G3b-2, so in a fresh process
# there is nothing to inherit, and G3b-5 is the only exact-equality (torch.equal) arm on the grouped path
# whose numerics this change alters. The surviving alternatives cannot stand in for it: G3b-1 runs at
# atol 2e-2, which is blind to a last-bit change, and G3b-4 exercises the flag-off path.
"$PYTHON" -m pytest tests/gpu/gpu_ci/test_grug_grouped_mm_parity.py tests/gpu/gpu_ci/test_grouped_gemm_parity.py \
  -k "g4a or g3b_1 or g3b_4 or g3b_5" -q -rA -s -p no:cacheprovider 2>&1 | tee one_gpu.log
one_gpu_status=${PIPESTATUS[0]}
echo "::: ONE-GPU PARITY EXIT=$one_gpu_status"

"$PYTHON" -m pytest tests/gpu/test_grug_fsdp2_train_eval_parity.py -q -rA -s -p no:cacheprovider 2>&1 | tee fsdp2.log
fsdp2_status=${PIPESTATUS[0]}
echo "::: FSDP2 PARITY EXIT=$fsdp2_status"

# An all-skip pytest exits 0, so a green status is not evidence that anything was measured. The GPU
# count above catches only one of the three ways require_hoppers skips (it also skips on a narrowed
# CUDA_VISIBLE_DEVICES and on compute capability below 9). Assert on the OUTCOME instead of
# re-deriving the preconditions, so this cannot drift from the helper it stands in for.
for stage in one_gpu fsdp2; do
  eval "status=\$${stage}_status"
  if [ "$status" -eq 0 ] && ! grep -qE "[0-9]+ passed" "$stage.log"; then
    echo "::: $stage MEASURED NOTHING -- every arm skipped, which exits 0"
    eval "${stage}_status=1"
  fi
done

# Every stage ran; now let the job's state say whether any of them failed.
[ "$probe_status" -eq 0 ] && [ "$one_gpu_status" -eq 0 ] && [ "$fsdp2_status" -eq 0 ]
