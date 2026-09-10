#!/usr/bin/env bash
set -euo pipefail

# CPython 3.12, Linux x86_64, H100. Requires git and a CUDA-supported C++ compiler.
package="${1:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}"
build_dir="$(realpath -m "${2:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
source_subdir=.
export MAX_JOBS=2
case "$package" in
    flash-attn)
        repository=Dao-AILab/flash-attention
        source_tag=v2.8.3
        source_commit=060c9188beec3a8b62b33a3bfa6d5d2d44975fab
        export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90
        ;;
    causal-conv1d)
        repository=Dao-AILab/causal-conv1d
        source_tag=v1.6.1.post4
        source_commit=9e4ace0b1d53ede275308abf25f64a1fc04c5fd4
        export CAUSAL_CONV1D_FORCE_BUILD=TRUE
        ;;
    mamba-ssm)
        repository=state-spaces/mamba
        source_tag=v2.3.1
        source_commit=c5afbdf3bda1a09d68f65181ae3a43ec71079820
        export MAMBA_FORCE_BUILD=TRUE
        ;;
    transformer-engine-torch)
        repository=NVIDIA/TransformerEngine
        source_tag=v2.11
        source_commit=c188b533cc3721ca9c6bbfd26148f5cf60108c25
        source_subdir=transformer_engine/pytorch
        export NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_NO_LOCAL_VERSION=1 NVTE_BUILD_MAX_JOBS=1
        export MAX_JOBS=1
        ;;
    *) echo "unsupported package: $package" >&2; exit 2 ;;
esac

mkdir -p "$build_dir"
uv venv --allow-existing --python 3.12.14 "$build_dir/venv"
uv pip sync --python "$build_dir/venv/bin/python" \
    --link-mode copy \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --index-strategy unsafe-best-match "$script_dir/native-cu132.txt"

if [[ ! -d "$build_dir/source" ]]; then
    git clone --depth 1 --branch "$source_tag" "https://github.com/$repository.git" "$build_dir/source"
fi
test "$(git -C "$build_dir/source" rev-parse HEAD)" = "$source_commit"
git -C "$build_dir/source" submodule update --init --recursive

export VIRTUAL_ENV="$build_dir/venv"
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
# NVIDIA's runtime wheel ships the SONAME but no development linker name.
ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/include:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/include"
export NVCC_THREADS=1
nvcc --version
c++ --version
uv build --wheel --no-build-isolation --python "$VIRTUAL_ENV/bin/python" \
    --out-dir "$build_dir/dist" "$build_dir/source/$source_subdir"
sha256sum "$build_dir"/dist/*.whl > "$build_dir/SHA256SUMS"
