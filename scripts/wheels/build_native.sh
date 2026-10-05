#!/usr/bin/env bash
set -euo pipefail

# CPython 3.12, Linux. Every new artifact needs qualification on its target GPU.
package="${1:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}"
build_dir="$(realpath -m "${2:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
native_local_version="$(python3.12 -c 'import json, sys; print(json.load(open(sys.argv[1]))["local_version"])' "$script_dir/native_versions.json")"
flash_attn_upstream_version="$(python3.12 -c 'import json, sys; print(json.load(open(sys.argv[1]))["flash_attn_upstream_version"])' "$script_dir/native_versions.json")"
source_subdir=.
source_patch=
python_version=3.12.14
architecture="$(uname -m)"
case "$architecture" in
    x86_64) cuda_architecture=90; torch_cuda_architecture=9.0 ;;
    aarch64) cuda_architecture=100; torch_cuda_architecture=10.0 ;;
    *) echo "unsupported build architecture: $architecture" >&2; exit 2 ;;
esac
max_jobs=2
package_environment=()
case "$package" in
    flash-attn)
        repository=Dao-AILab/flash-attention
        source_commit=a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2
        source_patch=flash-attn.patch
        package_environment+=(FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_LOCAL_VERSION= "FLASH_ATTN_CUDA_ARCHS=$cuda_architecture")
        ;;
    causal-conv1d)
        repository=Dao-AILab/causal-conv1d
        source_commit=cd81f0413cad2fc1e6f17e785ac39f59aae690cd
        source_patch=causal-conv1d.patch
        package_environment+=(CAUSAL_CONV1D_FORCE_BUILD=TRUE "CAUSAL_CONV1D_LOCAL_VERSION=$native_local_version")
        ;;
    mamba-ssm)
        repository=state-spaces/mamba
        source_commit=a14b1dff0454a3bc27d9eb31355dc01e4b2490ec
        source_patch=mamba-ssm.patch
        package_environment+=(MAMBA_FORCE_BUILD=TRUE "MAMBA_LOCAL_VERSION=$native_local_version")
        ;;
    transformer-engine-torch)
        repository=NVIDIA/TransformerEngine
        source_commit=5e52befd5262c06289106338c308079d6adb391f
        source_subdir=transformer_engine/pytorch
        package_environment+=(NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_NO_LOCAL_VERSION=1 NVTE_BUILD_MAX_JOBS=1)
        max_jobs=1
        ;;
    megatron-core)
        repository=NVIDIA/Megatron-LM
        source_commit=4b4acac9a1d28ea6829c8d4f566d75698a21249d
        source_patch=megatron-core.patch
        package_environment+=(NO_VCS_VERSION=1)
        ;;
    megatron-bridge)
        repository=NVIDIA-NeMo/Megatron-Bridge
        source_commit=c0e164ed2aedac4ad1c877780e2564a19d5d54ec
        source_patch=megatron-bridge.patch
        package_environment+=(NO_VCS_VERSION=1)
        ;;
    fast-hadamard-transform)
        repository=Dao-AILab/fast-hadamard-transform
        source_commit=e7706faf8d1c3b9f241e36860640ad1dac644ede
        package_environment+=(FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE)
        ;;
    *) echo "unsupported package: $package" >&2; exit 2 ;;
esac

mkdir -p "$build_dir"
uv venv --allow-existing --python "$python_version" "$build_dir/venv"
uv pip sync --python "$build_dir/venv/bin/python" \
    --link-mode copy \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --index-strategy unsafe-best-match "$script_dir/native-cu132.txt"

if [[ ! -d "$build_dir/source" ]]; then
    git init --quiet "$build_dir/source"
    git -C "$build_dir/source" remote add origin "https://github.com/$repository.git"
    git -C "$build_dir/source" fetch --depth 1 origin "$source_commit"
    git -C "$build_dir/source" checkout --quiet --detach FETCH_HEAD
fi
test "$(git -C "$build_dir/source" rev-parse HEAD)" = "$source_commit"
if [[ -n "$(git -C "$build_dir/source" status --porcelain --untracked-files=all --ignore-submodules=none)" ]]; then
    echo "Build source must be clean: $build_dir/source" >&2
    exit 1
fi
git -C "$build_dir/source" submodule update --init --recursive
git -C "$build_dir/source" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=all --ignore-submodules=none)"'
if [[ -n "$source_patch" ]]; then
    git -C "$build_dir/source" apply --unidiff-zero --check "$script_dir/patches/$source_patch"
    git -C "$build_dir/source" apply --unidiff-zero "$script_dir/patches/$source_patch"
fi

virtual_env="$build_dir/venv"
site_packages="$virtual_env/lib/python${python_version%.*}/site-packages"
cuda_home="$site_packages/nvidia/cu13"
# NVIDIA's runtime wheel ships the SONAME but no development linker name.
ln -sf libcudart.so.13 "$cuda_home/lib/libcudart.so"
build_environment=(
    "VIRTUAL_ENV=$virtual_env"
    "CUDA_HOME=$cuda_home"
    "PATH=$virtual_env/bin:$cuda_home/bin:$PATH"
    "CPATH=$site_packages/nvidia/cudnn/include:$site_packages/nvidia/nccl/include"
    "NVCC_THREADS=1"
    "MAX_JOBS=$max_jobs"
    "TORCH_CUDA_ARCH_LIST=$torch_cuda_architecture"
    "${package_environment[@]}"
)
{
    printf 'package=%s\narchitecture=%s\nsource=%s\n' "$package" "$architecture" "$source_commit"
    sha256sum "$0" "$script_dir/native-cu132.txt" "$script_dir/native_versions.json"
    if [[ -n "$source_patch" ]]; then
        sha256sum "$script_dir/patches/$source_patch"
        git -C "$build_dir/source" diff --binary
    fi
    git -C "$build_dir/source" submodule status --recursive
    "$virtual_env/bin/python" --version
    uv --version
    "$cuda_home/bin/nvcc" --version
    "$cuda_home/bin/ptxas" --version
    c++ --version
    ldd --version
    "$virtual_env/bin/python" -c \
        'import torch; print("torch=" + torch.__version__); print("cxx11_abi=" + str(torch.compiled_with_cxx11_abi()))'
} > "$build_dir/BUILD_INFO"
uv pip freeze --python "$virtual_env/bin/python" > "$build_dir/BUILD_REQUIREMENTS.txt"
cat "$build_dir/BUILD_INFO"
env "${build_environment[@]}" uv build --wheel --no-build-isolation --python "$virtual_env/bin/python" \
    --out-dir "$build_dir/dist" "$build_dir/source/$source_subdir"
if [[ "$package" == flash-attn ]]; then
    upstream_wheel="$build_dir/dist/flash_attn-$flash_attn_upstream_version-cp312-cp312-linux_$architecture.whl"
    source_sha256="$(sha256sum "$upstream_wheel" | cut -d ' ' -f 1)"
    "$virtual_env/bin/python" "$script_dir/retag_flash_attn.py" "$upstream_wheel" \
        --source-sha256 "$source_sha256" --output "$build_dir/dist" --proof "$build_dir/RETAG_PROOF.json"
    mkdir -p "$build_dir/upstream-wheel"
    mv "$upstream_wheel" "$build_dir/upstream-wheel/"
fi
(cd "$build_dir/dist" && sha256sum -- *.whl) > "$build_dir/SHA256SUMS"
