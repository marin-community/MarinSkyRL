#!/usr/bin/env bash
# The grouped-MoE combine gates, on two H100s. Run inside an Iris GPU task:
#
#   $IRIS --cluster marin job run --target-cluster <peer with room> \
#     --job-name <user>-fsdp2-parity --priority batch \
#     --cpu 32 --memory 200GB --disk 400GB --gpu H100x2 --enable-extra-resources \
#     --no-sync --timeout 5400 -- bash -c 'bash skyrl-train/ci/run_fsdp2_train_eval_parity.sh'
#
# Three stages, each printing its own exit line:
#   1. ci/probe_combine_order.py: does the summation order change the result, and did the old
#      scatter_add vary it between launches? One GPU, about a minute.
#   2. The one-GPU parity gates: the Grug grouped_mm gates and the parity arms of the Qwen
#      grouped-GEMM swap. Both paths use the fixed-order combine.
#   3. The FSDP2 train/eval parity gate: the real worker path on two GPUs, six arms, including a
#      26-layer arm at the production sequence length.
set -uo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-parity-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$ENV_DIR" production
# resolve_runtime.sh sets -e. Turn it off so a failing stage does not skip the later ones; the exit
# status is assembled at the end.
set +e
cd "$REPOSITORY_ROOT/skyrl-train"
# Ray's zip runtime flattens the repository symlink, so materialise the sibling package.
rm -rf skyrl-gym && cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Fail fast on a one-GPU node. The FSDP2 gate needs two.
gpu_count="$(nvidia-smi -L | wc -l)"
if [ "$gpu_count" -lt 2 ]; then
  echo "::: NEED 2 GPUs, found $gpu_count"
  exit 1
fi

# The frozen training environment has no pytest and no pip. Install pytest into it with uv so the
# tests import the same torch and skyrl_train the trainer uses.
uv pip install --quiet --python "$PYTHON" pytest
"$PYTHON" -c "import pytest, torch, ray; print(f'pytest {pytest.__version__} | torch {torch.__version__} | ray {ray.__version__}')"

"$PYTHON" ci/probe_combine_order.py
probe_status=$?
echo "::: COMBINE PROBE EXIT=$probe_status"

# No -x: each arm is a separate result, and -x would stop at the first failure.
# G3b-2 is excluded because it fails on main: it expects the router-replay controller to be gone after a
# grad-enabled forward, and the teardown deferral in model_wrapper.py keeps it. G3b-5 is the only arm on
# the grouped path that compares logits with torch.equal, so it is the one that sees a last-bit change.
"$PYTHON" -m pytest tests/gpu/gpu_ci/test_grug_grouped_mm_parity.py tests/gpu/gpu_ci/test_grouped_gemm_parity.py \
  -k "g4a or g3b_1 or g3b_4 or g3b_5" -q -rA -s -p no:cacheprovider 2>&1 | tee one_gpu.log
one_gpu_status=${PIPESTATUS[0]}
echo "::: ONE-GPU PARITY EXIT=$one_gpu_status"

"$PYTHON" -m pytest tests/gpu/test_grug_fsdp2_train_eval_parity.py -q -rA -s -p no:cacheprovider 2>&1 | tee fsdp2.log
fsdp2_status=${PIPESTATUS[0]}
echo "::: FSDP2 PARITY EXIT=$fsdp2_status"

# A pytest run where every test skips exits 0. require_hoppers skips on too few GPUs, a narrowed
# CUDA_VISIBLE_DEVICES, or compute capability below 9, so check that each stage passed at least one test.
for stage in one_gpu fsdp2; do
  eval "status=\$${stage}_status"
  if [ "$status" -eq 0 ] && ! grep -qE "[0-9]+ passed" "$stage.log"; then
    echo "::: $stage ran no tests: every arm skipped"
    eval "${stage}_status=1"
  fi
done

[ "$probe_status" -eq 0 ] && [ "$one_gpu_status" -eq 0 ] && [ "$fsdp2_status" -eq 0 ]
