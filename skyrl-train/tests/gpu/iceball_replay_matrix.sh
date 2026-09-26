#!/usr/bin/env bash
set -euo pipefail

cd /app
python marinskyrl/environment_contract.py write-frozen-cuda-runtime /tmp/iceball-replay-env
source /tmp/iceball-replay-env
if [[ ! -e "$CUDA_HOME/lib64" ]]; then
  ln -s "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
fi
ln -sf "$CUDA_HOME/lib/libcudart.so.13" .venv/lib/libcudart.so
ln -sf "$CUDA_HOME/lib/libnvrtc.so.13" .venv/lib/libnvrtc.so
export PYTHONPATH=/app
python -u skyrl-train/tests/gpu/iceball_replay_matrix.py "$@"
