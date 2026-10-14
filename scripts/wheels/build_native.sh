#!/usr/bin/env bash
set -euo pipefail

# CPython 3.12, Linux. Every new artifact needs qualification on its target GPU.
package="${1:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}"
build_dir="$(realpath -m "${2:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
python_version=3.12.14
if [[ "$package" == megatron-bridge || "$package" == speculators ]]; then
    python_version=3.12.13
fi
bootstrap_python="$(uv python find "$python_version")"
native_local_version="$("$bootstrap_python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["local_version"])' "$script_dir/native_versions.json")"
source_subdir=.
source_patches=()
architecture="$(uname -m)"
case "$architecture" in
    x86_64) cuda_architecture=90; torch_cuda_architecture=9.0 ;;
    aarch64) cuda_architecture=100; torch_cuda_architecture=10.0 ;;
    *) echo "unsupported build architecture: $architecture" >&2; exit 2 ;;
esac
max_jobs="${MAX_JOBS:-2}"
package_environment=()
case "$package" in
    flash-attn)
        repository=Dao-AILab/flash-attention
        source_patches+=(flash-attn.patch)
        # Upstream's native option sets both wheel and metadata versions.
        package_environment+=(FLASH_ATTENTION_FORCE_BUILD=TRUE "FLASH_ATTN_LOCAL_VERSION=$native_local_version" "FLASH_ATTN_CUDA_ARCHS=$cuda_architecture")
        ;;
    transformer-engine-torch)
        repository=NVIDIA/TransformerEngine
        source_subdir=transformer_engine/pytorch
        package_environment+=(NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_NO_LOCAL_VERSION=1 NVTE_BUILD_MAX_JOBS=1)
        max_jobs=1
        ;;
    megatron-core)
        repository=NVIDIA/Megatron-LM
        source_patches+=(megatron-core-optimizer-routes.patch megatron-core-fa4-import.patch megatron-core-frontend.patch megatron-core-version.patch)
        package_environment+=(NO_VCS_VERSION=1)
        ;;
    megatron-bridge)
        repository=NVIDIA-NeMo/Megatron-Bridge
        source_patches+=(megatron-bridge-flashinfer.patch megatron-bridge-version.patch)
        package_environment+=(NO_VCS_VERSION=1)
        ;;
    speculators)
        repository=marin-community/speculators
        source_patches+=(speculators-transformers.patch speculators-version.patch)
        package_environment+=(SPECULATORS_BUILD_TYPE=nightly SPECULATORS_BUILD_ITERATION=16)
        ;;
    fast-hadamard-transform)
        repository=Dao-AILab/fast-hadamard-transform
        package_environment+=(FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE)
        ;;
    *) echo "unsupported package: $package" >&2; exit 2 ;;
esac
source_commit="$("$bootstrap_python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["sources"][sys.argv[2]])' "$script_dir/native_versions.json" "$package")"

mkdir -p "$build_dir"
uv venv --allow-existing --python "$python_version" "$build_dir/venv"
build_requirements="$script_dir/native-cu132.txt"
build_index=https://download.pytorch.org/whl/cu132
if [[ "$package" == megatron-bridge ]]; then
    build_requirements="$script_dir/bridge-build.txt"
    build_index=https://download.pytorch.org/whl/cpu
elif [[ "$package" == speculators ]]; then
    build_requirements="$script_dir/speculators-build.txt"
    build_index=https://download.pytorch.org/whl/cpu
fi
uv pip sync --python "$build_dir/venv/bin/python" \
    --link-mode copy \
    --extra-index-url "$build_index" \
    --index-strategy unsafe-best-match "$build_requirements"

if [[ ! -d "$build_dir/source" ]]; then
    git init --quiet "$build_dir/source"
    git -C "$build_dir/source" remote add origin "https://github.com/$repository.git"
    git -C "$build_dir/source" fetch --depth 1 origin "$source_commit"
    git -C "$build_dir/source" checkout --quiet --detach FETCH_HEAD
fi
test "$(git -C "$build_dir/source" rev-parse HEAD)" = "$source_commit"
if [[ "$package" == speculators ]]; then
    git -C "$build_dir/source" fetch --depth 64 origin "$source_commit"
    git -C "$build_dir/source" fetch --depth 1 origin refs/tags/v0.8.0:refs/tags/v0.8.0
    test "$(git -C "$build_dir/source" rev-parse 'v0.8.0^{commit}')" = 894e71f96e0550c3b4cf3915fe6211aab09cadbb
    git -C "$build_dir/source" merge-base --is-ancestor 'v0.8.0^{commit}' HEAD
fi
if [[ -n "$(git -C "$build_dir/source" status --porcelain --untracked-files=all --ignore-submodules=none)" ]]; then
    echo "Build source must be clean: $build_dir/source" >&2
    exit 1
fi
git -C "$build_dir/source" submodule update --init --recursive
git -C "$build_dir/source" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=all --ignore-submodules=none)"'
for source_patch in "${source_patches[@]}"; do
    git -C "$build_dir/source" apply --unidiff-zero --check "$script_dir/patches/$source_patch"
    git -C "$build_dir/source" apply --unidiff-zero "$script_dir/patches/$source_patch"
done

virtual_env="$build_dir/venv"
site_packages="$virtual_env/lib/python${python_version%.*}/site-packages"
build_environment=(
    "VIRTUAL_ENV=$virtual_env"
    "PATH=$virtual_env/bin:$PATH"
    "NVCC_THREADS=1"
    "MAX_JOBS=$max_jobs"
    "TORCH_CUDA_ARCH_LIST=$torch_cuda_architecture"
    "${package_environment[@]}"
)
if [[ "$package" != megatron-bridge && "$package" != speculators ]]; then
    cuda_home="$site_packages/nvidia/cu13"
    # NVIDIA's runtime wheel ships the SONAME but no development linker name.
    ln -sf libcudart.so.13 "$cuda_home/lib/libcudart.so"
    build_environment+=(
        "CUDA_HOME=$cuda_home"
        "PATH=$virtual_env/bin:$cuda_home/bin:$PATH"
        "CPATH=$site_packages/nvidia/cudnn/include:$site_packages/nvidia/nccl/include"
    )
fi
{
    printf 'package=%s\narchitecture=%s\nsource=%s\n' "$package" "$architecture" "$source_commit"
    sha256sum "$0" "$build_requirements" "$script_dir/native_versions.json"
    for source_patch in "${source_patches[@]}"; do
        sha256sum "$script_dir/patches/$source_patch"
    done
    if [[ ${#source_patches[@]} -gt 0 ]]; then
        git -C "$build_dir/source" diff --binary
    fi
    git -C "$build_dir/source" submodule status --recursive
    "$virtual_env/bin/python" --version
    uv --version
    if [[ "$package" != megatron-bridge && "$package" != speculators ]]; then
        "$cuda_home/bin/nvcc" --version
        "$cuda_home/bin/ptxas" --version
        c++ --version
        ldd --version
    fi
    if [[ "$package" != speculators ]]; then
        "$virtual_env/bin/python" -c \
            'import torch; print("torch=" + torch.__version__); print("cxx11_abi=" + str(torch.compiled_with_cxx11_abi()))'
    fi
} > "$build_dir/BUILD_INFO"
uv pip freeze --python "$virtual_env/bin/python" > "$build_dir/BUILD_REQUIREMENTS.txt"
cat "$build_dir/BUILD_INFO"
env "${build_environment[@]}" uv build --wheel --no-build-isolation --python "$virtual_env/bin/python" \
    --out-dir "$build_dir/dist" "$build_dir/source/$source_subdir"
(cd "$build_dir/dist" && sha256sum -- *.whl) > "$build_dir/SHA256SUMS"
