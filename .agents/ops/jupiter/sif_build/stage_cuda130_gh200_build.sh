#!/bin/bash
# Stage immutable source inputs for the Jupiter CUDA 13.0 / GH200 image build.
# This script performs network I/O on a login node. It does not compile code or
# create an Apptainer image; build_cuda130_gh200.sbatch performs that work on a
# compute node.
set -euo pipefail

JUPITER_ROOT=${JUPITER_ROOT:-/e/scratch/jureap59/feuer1}
STAGING_ROOT=${STAGING_ROOT:-$JUPITER_ROOT/sif_build}
BASE_SIF=${BASE_SIF:-$JUPITER_ROOT/containers/skyrl_megatron_vllm0202rc0_r5.sif}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
VLLM_REPOSITORY=https://github.com/marin-community/vllm.git
VLLM_REF=${VLLM_REF:-refs/heads/gpu}
HARBOR_REPOSITORY=https://github.com/marin-community/harbor.git
HARBOR_REF=${HARBOR_REF:-refs/heads/main}
NCCL_VERSION=2.31.2
NCCL_WHEEL=nvidia_nccl_cu13-2.31.2-py3-none-manylinux_2_18_aarch64.whl
NCCL_URL=https://files.pythonhosted.org/packages/52/a0/530efd7db8857c0436868bb7df9764f09fde2bd4d1f0bae546eec9fc40d0/$NCCL_WHEEL
NCCL_SHA256=b5563f8e2534f363d93ace022670ba016d3717e190ac4eba564d05fbbe8495b1

require_file() {
  if [[ ! -f $1 ]]; then
    echo "FATAL: required file is missing: $1" >&2
    exit 2
  fi
}

clone_ref() {
  local repository=$1
  local ref=$2
  local destination=$3
  local recursive=${4:-false}

  git init --quiet "$destination"
  git -C "$destination" remote add origin "$repository"
  git -C "$destination" fetch --quiet --depth 1 origin "$ref"
  git -C "$destination" checkout --quiet --detach FETCH_HEAD
  if [[ $recursive == true ]]; then
    git -C "$destination" submodule update --init --recursive --depth 1
  fi
}

require_file "$BASE_SIF"
if [[ $(uname -m) != aarch64 ]]; then
  echo "FATAL: stage this build on a Jupiter aarch64 login node" >&2
  exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
bundle=$STAGING_ROOT/cuda130-gh200-$stamp
local_stage=$(mktemp -d "${TMPDIR:-/tmp}/marinskyrl-sif-stage.XXXXXX")
sources=$local_stage/sources

cleanup() {
  case "$local_stage" in
    /tmp/marinskyrl-sif-stage.*|/var/tmp/marinskyrl-sif-stage.*) rm -rf -- "$local_stage" ;;
    *) echo "WARNING: refusing to clean unexpected staging path: $local_stage" >&2 ;;
  esac
}
trap cleanup EXIT

mkdir -p "$STAGING_ROOT/logs" "$bundle/wheels" "$sources"
cp "$SCRIPT_DIR/validate_cuda130_gh200.py" "$bundle/validate_cuda130_gh200.py"

echo "Staging source trees in login-local storage before archiving them to $bundle"
clone_ref "$VLLM_REPOSITORY" "$VLLM_REF" "$sources/vllm"
clone_ref "$HARBOR_REPOSITORY" "$HARBOR_REF" "$sources/harbor"

# These revisions come from the selected vLLM source tree. Keep them in sync
# with its CMake files when changing VLLM_REF. Fail before cloning large source
# trees if the moving branch has changed any of them.
declare -A vllm_dependency_markers=(
  [cutlass]=v4.4.2
  [vllm-flash-attn]=f3e1a4f74c99145c0717709860bf765de1703779
  [flashmla]=a8f794d1251cbfd88a5011445dd5582289c727e4
  [triton]=v3.5.1
  [deepgemm]=e21c821f39a2056d68067a466c64ddc942200106
  [fmha-sm100]=087c161814d4d9c735b46c21212a09e5f8eb92fa
  [flashkda]=b5d11010ff01c1d4a683c0dde42e76cbeaa8107f
  [qutlass]=e74319e3405ce6d71965732880f5dc1f52371f64
  [tml-fa4]=b206834606ed5b5f21f8eed6b0683f528ea9cf7d
)
for dependency in "${!vllm_dependency_markers[@]}"; do
  marker=${vllm_dependency_markers[$dependency]}
  if ! grep -RqsF "$marker" "$sources/vllm/CMakeLists.txt" "$sources/vllm/cmake/external_projects"; then
    echo "FATAL: vLLM changed its $dependency revision; update this staging script after review" >&2
    exit 2
  fi
done

clone_ref https://github.com/nvidia/cutlass.git refs/tags/v4.4.2 "$sources/cutlass"
clone_ref https://github.com/vllm-project/flash-attention.git f3e1a4f74c99145c0717709860bf765de1703779 \
  "$sources/vllm-flash-attn" true
clone_ref https://github.com/vllm-project/FlashMLA.git a8f794d1251cbfd88a5011445dd5582289c727e4 \
  "$sources/flashmla" true
clone_ref https://github.com/triton-lang/triton.git refs/tags/v3.5.1 "$sources/triton"
clone_ref https://github.com/vllm-project/DeepGEMM.git e21c821f39a2056d68067a466c64ddc942200106 \
  "$sources/deepgemm" true
clone_ref https://github.com/vllm-project/MSA.git 087c161814d4d9c735b46c21212a09e5f8eb92fa \
  "$sources/fmha-sm100" true
clone_ref https://github.com/vllm-project/FlashKDA.git b5d11010ff01c1d4a683c0dde42e76cbeaa8107f \
  "$sources/flashkda" true
clone_ref https://github.com/IST-DASLab/qutlass.git e74319e3405ce6d71965732880f5dc1f52371f64 \
  "$sources/qutlass" true
clone_ref https://github.com/vllm-project/tml-fa4.git b206834606ed5b5f21f8eed6b0683f528ea9cf7d \
  "$sources/tml-fa4" true

wget --quiet "$NCCL_URL" -O "$bundle/wheels/$NCCL_WHEEL"
echo "$NCCL_SHA256  $bundle/wheels/$NCCL_WHEEL" | sha256sum --check

# Harbor uses uv_build. Stage its build backend so the compute-node build can
# run with network access disabled.
apptainer exec --pwd / "$BASE_SIF" python -m pip download \
  --disable-pip-version-check --no-deps --only-binary=:all: \
  --dest "$bundle/wheels" \
  'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' \
  'semantic-version>=2.8.2' 'uv_build>=0.8.4,<0.9.0'

vllm_commit=$(git -C "$sources/vllm" rev-parse HEAD)
harbor_commit=$(git -C "$sources/harbor" rev-parse HEAD)
base_sha256=$(sha256sum "$BASE_SIF" | awk '{print $1}')
cat > "$bundle/manifest.env" <<EOF
BASE_SIF=$BASE_SIF
BASE_SIF_SHA256=$base_sha256
VLLM_REPOSITORY=$VLLM_REPOSITORY
VLLM_REF=$VLLM_REF
VLLM_COMMIT=$vllm_commit
HARBOR_REPOSITORY=$HARBOR_REPOSITORY
HARBOR_REF=$HARBOR_REF
HARBOR_COMMIT=$harbor_commit
NCCL_VERSION=$NCCL_VERSION
NCCL_WHEEL=$NCCL_WHEEL
NCCL_SHA256=$NCCL_SHA256
CUDA_VERSION=13.0
CUDA_ARCH=9.0
EOF

find "$sources" -mindepth 1 -maxdepth 1 -type d -print0 \
  | sort -z \
  | while IFS= read -r -d '' source; do
      printf '%s=%s\n' "$(basename "$source")" "$(git -C "$source" rev-parse HEAD)"
    done > "$bundle/source-revisions.txt"
tar -C "$local_stage" -cf "$bundle/sources.tar" sources
sha256sum "$bundle/wheels"/* > "$bundle/wheel-sha256.txt"
sha256sum "$bundle/sources.tar" > "$bundle/sources.tar.sha256"
chmod -R a-w "$bundle/sources.tar" "$bundle/sources.tar.sha256" "$bundle/wheels" "$bundle/manifest.env" \
  "$bundle/source-revisions.txt" "$bundle/wheel-sha256.txt" "$bundle/validate_cuda130_gh200.py"

echo "Staging complete. Submit the compute-node build with:"
echo "sbatch --export=ALL,BUNDLE_DIR=$bundle $(dirname "$0")/build_cuda130_gh200.sbatch"
