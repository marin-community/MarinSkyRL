#!/usr/bin/env bash
set -euo pipefail

cd /app
python marinskyrl/environment_contract.py write-frozen-cuda-runtime /tmp/iceball-muonh-env
source /tmp/iceball-muonh-env
if [[ ! -e "$CUDA_HOME/lib64" ]]; then
  ln -s "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
fi
ln -sf "$CUDA_HOME/lib/libcudart.so.13" .venv/lib/libcudart.so
ln -sf "$CUDA_HOME/lib/libnvrtc.so.13" .venv/lib/libnvrtc.so
export PYTHONPATH=/app:/app/skyrl-train
uv pip install --python /app/.venv/bin/python pytest==9.1.1
if [[ "$#" -eq 0 ]]; then
  set -- skyrl-train/tests/gpu/test_grug_megatron_muonh.py
fi
python -m pytest -q "$@"
