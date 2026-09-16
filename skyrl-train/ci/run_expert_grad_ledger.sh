#!/usr/bin/env bash
# A11 -- the expert-gradient ledger, on one H100, inside an Iris job.
# Does select_backward on the stacked expert bank account for policy_backward? (O4)
# Builds the same frozen runtime the nightly and the A3 gate use, then runs the bench.
set -euo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-ledger-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$ENV_DIR" production

cd "$REPOSITORY_ROOT/skyrl-train"
# Ray's zip runtime flattens the sibling symlink; materialise it before setting source paths.
rm -rf skyrl-gym && cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "::: GPU"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PYTHON" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

# `set -e` would abort on a failing arm, so a status line placed after one could only ever print 0 --
# a health signal true by construction. Each arm's failure IS its result here, so the guard is off
# across the arms and each status is captured where it is still readable. The script exits non-zero
# if any arm failed, so a caller still learns about it.
set +e
ledger_status=0

echo "::: A11 main -- E sweep + grouped, default allocator"
"$PYTHON" ci/bench_expert_grad_ledger.py --experts 32 64 128 256 --profile-at 256
status=$?; [ "$status" -eq 0 ] || ledger_status=$status
echo "::: A11-MAIN EXIT=$status"

# O4 is token-INDEPENDENT by construction. Halving rows/expert must NOT halve the backward.
echo "::: A11 token-count control -- 112 -> 56 rows/expert, sliced vs separate only"
"$PYTHON" ci/bench_expert_grad_ledger.py --experts 256 --arms sliced separate --rows-per-expert 56
status=$?; [ "$status" -eq 0 ] || ledger_status=$status
echo "::: A11-TOKENS EXIT=$status"

# H4 control: if expandable_segments moves the answer, the number is fragmentation, not bandwidth.
echo "::: A11 allocator control -- expandable_segments (the run's own setting, F6's fix)"
# ⚠️ The env assignment and its command are ONE statement. An earlier edit inserted a comment between
# the trailing backslash and this line, which left the benchmark running WITHOUT the allocator
# setting -- silently turning the control into a duplicate of the arm above it. `bash -n` passes on
# that, because it is valid shell and wrong.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" ci/bench_expert_grad_ledger.py --experts 256 --arms sliced separate
status=$?; [ "$status" -eq 0 ] || ledger_status=$status
echo "::: A11-ALLOC EXIT=$status"
exit "$ledger_status"
