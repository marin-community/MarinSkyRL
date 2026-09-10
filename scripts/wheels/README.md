# CUDA 13.2 native wheels

`build_native.sh` builds the existing FlashAttention, causal-conv1d, Mamba, and
Transformer Engine Torch releases against Torch 2.13.0+cu132. It checks each
upstream source commit and builds locally instead of downloading an upstream
wheel compiled for a different Torch version. `native-cu132.txt` pins the build
environment, including CUDA 13.2.1's NVCC and CCCL packages.

Use CPython 3.12.14 on Linux x86_64 with git, a C++ compiler, and uv. The H100
builds used the Iris task image
`ghcr.io/marin-community/iris-task@sha256:72c0074ff00cc8b361b6fe35bc67c1f057bfa0cc31b09aba2edfb28050e5a386`,
GCC/G++ `14.2.0-19`, glibc `2.41-12+deb13u3`, git `1:2.47.3-0+deb13u1`, and
uv `0.10.3`. Install the compiler and git inside the build container. These are
Linux wheels for that task environment; they do not claim manylinux portability.

```bash
scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn
scripts/wheels/build_native.sh causal-conv1d /tmp/build-causal-conv1d
scripts/wheels/build_native.sh mamba-ssm /tmp/build-mamba
scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te
```

Each directory retains its environment and source build cache. Wheels are written
to `dist/`, with their SHA-256 digests in `SHA256SUMS`. FlashAttention targets SM90;
the other projects retain their upstream architecture choices. No aarch64 wheel
is built by this script.

Before publishing a wheel, install its exact bytes in the proposed runtime and
run its native forward and backward checks on H100. Publish the source commits,
build environment, and checksums with the wheel assets. Adoption URLs and wheel
hashes belong in the root dependency manifest and lock.
