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

echo "::: A11 main -- E sweep + grouped, default allocator"
"$PYTHON" ci/bench_expert_grad_ledger.py --experts 32 64 128 256 --profile-at 256
echo "::: A11-MAIN EXIT=$?"

# O4 is token-INDEPENDENT by construction. Halving rows/expert must NOT halve the backward.
echo "::: A11 token-count control -- 112 -> 56 rows/expert, sliced vs separate only"
"$PYTHON" ci/bench_expert_grad_ledger.py --experts 256 --arms sliced separate --rows-per-expert 56
echo "::: A11-TOKENS EXIT=$?"

# H4 control: if expandable_segments moves the answer, the number is fragmentation, not bandwidth.
echo "::: A11 allocator control -- expandable_segments (the run's own setting, F6's fix)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" ci/bench_expert_grad_ledger.py --experts 256 --arms sliced separate
echo "::: A11-ALLOC EXIT=$?"
