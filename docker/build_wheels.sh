#!/usr/bin/env bash
#
# build_wheels.sh — build the SLOW from-source CUDA/x86 wheels for gpu-rl ONCE
# and persist them to docker/wheelhouse/, so subsequent `gpu-rl` image rebuilds
# install the prebuilt wheels instead of recompiling (no nvcc on a rebuild).
#
# The two heavy nvcc compiles (the reason a gpu-rl rebuild is heavy) are:
#   - the vLLM FORK (cutlass GEMM kernels)
#   - flash-attn (its flash_attn_2_cuda extension)
# Both are pinned to the exact fork commit / flash-attn version / torch 2.11 /
# CUDA 12.8 / cp312 / x86_64 ABI declared as ARGs in docker/Dockerfile.gpu-rl.
#
# WHAT THIS DOES
#   1. Builds ONLY the `wheel-builder` Dockerfile stage (the nvcc compiles).
#   2. Exports /wheels/*.whl + /wheels/MANIFEST from that stage to
#      docker/wheelhouse/ (gitignored — large binaries).
#   3. After this runs once, `docker/build_and_push.sh gpu-rl` (or the buildx
#      command in docker/README.gpu-rl-wheelcache.md) installs THOSE wheels.
#
# CACHE KEY (what invalidates the wheels) — recomputed from the Dockerfile ARGs:
#   { VLLM_FORK_COMMIT, FLASH_ATTN_VERSION, torch 2.11.0, CUDA 12.8, cp312,
#     x86_64, TORCH_CUDA_ARCH_LIST "8.0;9.0" }
# Re-run this script ONLY when one of those pins changes (it overwrites the
# wheelhouse + MANIFEST). torchtitan is pure-python and is NOT wheel-cached.
#
# WHERE TO RUN
#   On an x86_64 (linux/amd64) host with Docker + nvcc-capable buildkit and
#   enough RAM (MAX_JOBS=8 * ~5GB/nvcc-job => budget ~40GB). The production
#   gpu-rl image is amd64-only, so build amd64 wheels. On an arm64 Mac this would
#   run under QEMU (impractically slow + RAM-bound) — use a real x86 build host
#   / GPU build pod / CI x86 runner.
#
# USAGE
#   ./docker/build_wheels.sh                 # build wheels -> docker/wheelhouse/
#   PLATFORM=linux/amd64 ./docker/build_wheels.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.gpu-rl"
WHEELHOUSE="$SCRIPT_DIR/wheelhouse"

# The production gpu-rl image is linux/amd64 ONLY (CoreWeave H100 / x86 CUDA).
PLATFORM="${PLATFORM:-linux/amd64}"
BUILDER_NAME="${BUILDER_NAME:-openthoughts-builder}"

echo "============================================"
echo "gpu-rl wheelhouse build"
echo "============================================"
echo "Dockerfile:  $DOCKERFILE"
echo "Wheelhouse:  $WHEELHOUSE  (gitignored)"
echo "Platform:    $PLATFORM"
echo "Builder:     $BUILDER_NAME"
echo "============================================"

cd "$REPO_ROOT"

# Ensure a buildx builder exists (the wheel-builder stage needs buildkit).
if ! docker buildx inspect "$BUILDER_NAME" &>/dev/null; then
    echo ">>> Creating buildx builder $BUILDER_NAME ..."
    docker buildx create --name "$BUILDER_NAME" --use --bootstrap
else
    docker buildx use "$BUILDER_NAME"
fi

mkdir -p "$WHEELHOUSE"

# Build ONLY the wheel-builder stage and export /wheels to the host wheelhouse.
# --output type=local copies the stage's `/wheels` dir to $WHEELHOUSE on the host
# (we re-root the stage's filesystem at /wheels via the final scratch export).
echo ""
echo ">>> Building wheel-builder stage + exporting wheels to $WHEELHOUSE ..."
docker buildx build \
    --platform "$PLATFORM" \
    -f "$DOCKERFILE" \
    --target wheel-builder \
    --build-arg WHEEL_SOURCE=wheel-builder \
    --output "type=local,dest=$SCRIPT_DIR/.wheelbuild-export" \
    .

# The wheel-builder stage holds the wheels at /wheels; the local export mirrors
# the whole stage fs, so copy just the wheels + manifest into the wheelhouse.
echo ""
echo ">>> Collecting wheels from export ..."
find "$SCRIPT_DIR/.wheelbuild-export/wheels" -maxdepth 1 -name '*.whl' -exec cp -v {} "$WHEELHOUSE/" \;
if [[ -f "$SCRIPT_DIR/.wheelbuild-export/wheels/MANIFEST" ]]; then
    cp -v "$SCRIPT_DIR/.wheelbuild-export/wheels/MANIFEST" "$WHEELHOUSE/MANIFEST"
fi
rm -rf "$SCRIPT_DIR/.wheelbuild-export"

echo ""
echo "============================================"
echo "Wheelhouse contents:"
ls -la "$WHEELHOUSE"
echo "============================================"
echo "Done. Now build the image with the cached wheels:"
echo "  ./docker/build_and_push.sh gpu-rl"
echo "(the rl stage COPYs docker/wheelhouse/*.whl and installs them — no nvcc)"
