#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=mskyrl-sif-cu130
#SBATCH --output=/e/scratch/jureap59/feuer1/sif_build/logs/cuda130-gh200-%j.log

# Compile vLLM and create the SIF on a Jupiter GH200 compute node. Network I/O
# is deliberately disabled for Python and CMake dependency resolution: all
# source trees and wheels must have been produced by stage_cuda130_gh200_build.sh.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_JOB_NODELIST:-} ]]; then
  echo "FATAL: this script must run as a Slurm compute job" >&2
  exit 2
fi
if [[ -z ${BUNDLE_DIR:-} || ! -f $BUNDLE_DIR/manifest.env || ! -f $BUNDLE_DIR/sources.tar ]]; then
  echo "FATAL: BUNDLE_DIR must name a staged build bundle" >&2
  exit 2
fi

JUPITER_ROOT=${JUPITER_ROOT:-/e/scratch/jureap59/feuer1}
CONTAINER_DIR=$JUPITER_ROOT/containers
source "$BUNDLE_DIR/manifest.env"

actual_base_sha256=$(sha256sum "$BASE_SIF" | awk '{print $1}')
if [[ $actual_base_sha256 != "$BASE_SIF_SHA256" ]]; then
  echo "FATAL: base SIF changed after staging" >&2
  exit 2
fi
module load Apptainer 2>/dev/null || true
command -v apptainer >/dev/null
mkdir -p "$JUPITER_ROOT/sif_build/logs" "$CONTAINER_DIR"

work_root=${SLURM_TMPDIR:-/tmp/marinskyrl-sif-$SLURM_JOB_ID}
case "$work_root" in
  /tmp/*|/var/tmp/*) ;;
  *) echo "FATAL: refusing non-local build root: $work_root" >&2; exit 2 ;;
esac
mkdir -p "$work_root" "$work_root/apptainer-tmp" "$work_root/apptainer-cache"
export APPTAINER_TMPDIR=$work_root/apptainer-tmp
export APPTAINER_CACHEDIR=$work_root/apptainer-cache

build_inputs=$work_root/inputs
sandbox=$work_root/sandbox
wheelhouse=$work_root/wheelhouse
mkdir -p "$wheelhouse"
sha256sum --check "$BUNDLE_DIR/sources.tar.sha256"
tar -C "$work_root" -xf "$BUNDLE_DIR/sources.tar"
mv "$work_root/sources" "$build_inputs"
cp -a "$BUNDLE_DIR/wheels/." "$wheelhouse/"
chmod -R u+w "$build_inputs" "$wheelhouse"
if [[ $(git -C "$build_inputs/vllm" rev-parse HEAD) != "$VLLM_COMMIT" ]]; then
  echo "FATAL: staged vLLM revision does not match the manifest" >&2
  exit 2
fi
if [[ $(git -C "$build_inputs/harbor" rev-parse HEAD) != "$HARBOR_COMMIT" ]]; then
  echo "FATAL: staged Harbor revision does not match the manifest" >&2
  exit 2
fi

vllm_short=${VLLM_COMMIT:0:12}
output_name=marinskyrl_cuda130_gh200_vllm-gpu-${vllm_short}_nccl-2.31.2.sif
output_sif=$CONTAINER_DIR/$output_name
temporary_sif=$work_root/$output_name
if [[ -e $output_sif ]]; then
  echo "FATAL: output already exists: $output_sif" >&2
  exit 2
fi

echo "=== Build identity ==="
echo "job=$SLURM_JOB_ID node=$(hostname) bundle=$BUNDLE_DIR"
cat "$BUNDLE_DIR/manifest.env"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo "=== Convert the immutable r5 input to a compute-local sandbox ==="
apptainer build --sandbox "$sandbox" "$BASE_SIF"

echo "=== Build vLLM and install the requested revisions ==="
apptainer exec --writable --nv --pwd / \
  --bind "$build_inputs:/build/sources" \
  --bind "$wheelhouse:/build/wheels" \
  --env VLLM_COMMIT="$VLLM_COMMIT" \
  --env HARBOR_COMMIT="$HARBOR_COMMIT" \
  --env NCCL_VERSION="$NCCL_VERSION" \
  "$sandbox" bash -lc '
    set -euo pipefail
    test "$(uname -m)" = aarch64
    test "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" = 9.0
    test "$(python -c "import torch; print(torch.version.cuda)")" = 13.0

    export PIP_NO_INDEX=1
    export CMAKE_BUILD_TYPE=Release
    export CMAKE_ARGS=-DFETCHCONTENT_FULLY_DISCONNECTED=ON
    export MAX_JOBS=32
    export NVCC_THREADS=2
    export TORCH_CUDA_ARCH_LIST=9.0
    export VLLM_TARGET_DEVICE=cuda
    export VLLM_CUTLASS_SRC_DIR=/build/sources/cutlass
    export VLLM_FLASH_ATTN_SRC_DIR=/build/sources/vllm-flash-attn
    export FLASH_MLA_SRC_DIR=/build/sources/flashmla
    export TRITON_KERNELS_SRC_DIR=/build/sources/triton/python/triton_kernels/triton_kernels
    export DEEPGEMM_SRC_DIR=/build/sources/deepgemm
    export FMHA_SM100_SRC_DIR=/build/sources/fmha-sm100
    export FLASH_KDA_SRC_DIR=/build/sources/flashkda
    export QUTLASS_SRC_DIR=/build/sources/qutlass
    export TML_FA4_SRC_DIR=/build/sources/tml-fa4

    python -m pip install --disable-pip-version-check --no-deps --force-reinstall \
      "/build/wheels/nvidia_nccl_cu13-${NCCL_VERSION}-py3-none-manylinux_2_18_aarch64.whl"
    python -m pip install --disable-pip-version-check --no-deps --force-reinstall \
      /build/wheels/setuptools-*.whl /build/wheels/setuptools_scm-*.whl \
      /build/wheels/setuptools_rust-*.whl /build/wheels/semantic_version-*.whl

    cd /build/sources/vllm
    python use_existing_torch.py --prefix
    rm -rf .deps/*-build .deps/*-subbuild build
    SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0+marin.${VLLM_COMMIT:0:12}.cu130" \
      python -m pip wheel --disable-pip-version-check --no-build-isolation --no-deps \
      --wheel-dir /build/wheels .
    python -m pip install --disable-pip-version-check --no-deps --force-reinstall \
      /build/wheels/vllm-*.whl

    python -m pip install --disable-pip-version-check --no-deps /build/wheels/uv_build-*.whl
    python -m pip wheel --disable-pip-version-check --no-build-isolation --no-deps \
      --wheel-dir /build/wheels /build/sources/harbor
    python -m pip install --disable-pip-version-check --no-deps --force-reinstall \
      /build/wheels/harbor-*.whl

    mkdir -p /opt/marinskyrl-build
    printf "%s\n" "$VLLM_COMMIT" > /opt/marinskyrl-build/vllm-commit
    printf "%s\n" "$HARBOR_COMMIT" > /opt/marinskyrl-build/harbor-commit
    printf "%s\n" "$NCCL_VERSION" > /opt/marinskyrl-build/nccl-version
    python -m pip freeze > /opt/marinskyrl-build/pip-freeze.txt
  '

echo "=== Validate the writable sandbox ==="
apptainer exec --writable --nv --pwd / "$sandbox" \
  python "$BUNDLE_DIR/validate_cuda130_gh200.py" \
  --vllm-commit "$VLLM_COMMIT" --harbor-commit "$HARBOR_COMMIT"

echo "=== Build and validate the immutable SIF ==="
apptainer build "$temporary_sif" "$sandbox"
apptainer exec --nv --pwd / "$temporary_sif" \
  python "$BUNDLE_DIR/validate_cuda130_gh200.py" \
  --vllm-commit "$VLLM_COMMIT" --harbor-commit "$HARBOR_COMMIT"

output_sha256=$(sha256sum "$temporary_sif" | awk '{print $1}')
printf '%s  %s\n' "$output_sha256" "$output_name" > "$temporary_sif.sha256"
cp "$BUNDLE_DIR/manifest.env" "$temporary_sif.manifest.env"
mv "$temporary_sif" "$output_sif"
mv "$temporary_sif.sha256" "$output_sif.sha256"
mv "$temporary_sif.manifest.env" "$output_sif.manifest.env"

echo "=== Build complete ==="
echo "SIF=$output_sif"
cat "$output_sif.sha256"
