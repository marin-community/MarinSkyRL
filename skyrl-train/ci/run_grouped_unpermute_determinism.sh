#!/usr/bin/env bash
# F25: is the grouped-MM MoE combine nondeterministic? One H100, ~1 minute of compute.
# Mirrors run_expert_grad_ledger.sh: resolve the production runtime, then run the probe.
set -euo pipefail
REPOSITORY_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$REPOSITORY_ROOT/.iris-detprobe-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$ENV_DIR" production
cd "$REPOSITORY_ROOT/skyrl-train"
# Zip layers flatten symlinks, so the sibling package must be copied in (run_h100.sh:44-50).
rm -rf skyrl-gym && cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$PYTHON" ci/probe_grouped_unpermute_determinism.py
echo "::: PROBE EXIT=$?"
