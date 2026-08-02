#!/usr/bin/env bash

set -euo pipefail

report=${1:?pip-check report path is required}
platform=${2:?wheel platform is required}
install_megatron=${3:?Megatron selector is required}

grep -E '^The package `aiobotocore` requires `botocore[^`]*`, but `[^`]+` is installed$' "$report"
grep -E '^The package `gcsfs` requires `fsspec[^`]*`, but `[^`]+` is installed$' "$report"
grep -E '^The package `quack-kernels` requires `nvidia-cutlass-dsl[^`]*`, but `[^`]+` is installed$' "$report"

expected=3
case "$platform" in
  linux_x86_64) ;;
  linux_aarch64)
    grep -E '^The package `nvidia-cusparselt-cu12` was built for a different platform$' "$report"
    expected=4
    ;;
  *)
    echo "unsupported GPU-RL wheel platform: $platform" >&2
    exit 2
    ;;
esac

if [[ "$install_megatron" == "1" ]]; then
  grep -E '^The package `megatron-bridge` requires `nvidia-resiliency-ext`, but it.s not installed$' "$report"
  grep -E '^The package `megatron-bridge` requires `flashinfer-python[^`]*`, but `[^`]+` is installed$' "$report"
  grep -E '^The package `megatron-bridge` requires `flashinfer-cubin[^`]*`, but `[^`]+` is installed$' "$report"
  grep -E '^The package `transformer-engine-cu12` was built for a different platform$' "$report"
  expected=$((expected + 4))
  if [[ "$platform" == "linux_x86_64" ]]; then
    grep -E '^The package `transformer-engine-torch` requires `transformer-engine-cu13[^`]*`, but it.s not installed$' "$report"
    expected=$((expected + 1))
  fi
elif [[ "$install_megatron" != "0" ]]; then
  echo "INSTALL_MEGATRON must be 0 or 1, got: $install_megatron" >&2
  exit 2
fi

test "$(grep -c '^The package `' "$report")" = "$expected"
