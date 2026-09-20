# CUDA 13.2 native wheels

`build_native.sh` builds the existing FlashAttention, causal-conv1d, Mamba, and
Transformer Engine Torch releases against Torch 2.13.0+cu132. It checks each
upstream source commit and builds locally instead of downloading an upstream
wheel compiled for a different Torch version. `native-cu132.txt` pins the build
environment, including CUDA 13.2.1's NVCC and CCCL packages.

This checks in and tightens the build process first published with the
[original CUDA 13.2 wheels](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-torch2.13-20260910-6d12d7a0).
Reproducible here means that the source, dependencies, and build commands are
pinned. Independent builds are not guaranteed to produce identical bytes, so
runtime dependencies pin each published wheel by URL and SHA-256.

`build_native.sh` is a manual release tool. CI and runtime installation do not
invoke it; they consume the published, hash-pinned wheels.

FlashAttention uses upstream commit
[`4219765`](https://github.com/Dao-AILab/flash-attention/commit/4219765dfdd8913bfe26134f748dd5ffcedd3c39),
the merged FA2 and FA4 namespace-coexistence fix. That 2.8.4 source excludes the
bundled `flash_attn.cute` package written for an older CUTLASS DSL while retaining
FA2's native interface. The script fetches every source by its exact commit so a
moving branch or tag cannot change the input tree.

Use CPython 3.12.14 on Linux with git, a C++ compiler, and uv. Build on the
target CPU architecture: x86_64 for the H100 runtime or aarch64 for GB200. The
qualified FlashAttention 2.8.4 build used the Iris task image
`ghcr.io/marin-community/iris-task@sha256:ecdb2f7f90f8760a7e74c49b49b67d7ecf44557298860411c148c186706067f2`,
GCC/G++ `14.2.0-19`, glibc `2.41-12+deb13u3`, git `1:2.47.3-0+deb13u1`, and
uv `0.10.3`. Install the compiler and git inside the build container. These are
Linux wheels for that task environment; they do not claim manylinux portability.

```bash
scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn
scripts/wheels/build_native.sh causal-conv1d /tmp/build-causal-conv1d
scripts/wheels/build_native.sh mamba-ssm /tmp/build-mamba
scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te
# On an aarch64 GB200 task:
scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te 100
scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn 100
```

Each directory retains its environment and source build cache. Wheels are written
to `dist/`, with their SHA-256 digests in `SHA256SUMS`. The optional third
argument selects SM90 (the default) or SM100 for FlashAttention and Transformer
Engine. The wheel tag reflects the host CPU architecture. Causal-conv1d and
Mamba retain their upstream architecture choices.
Set `NATIVE_BUILD_MAX_JOBS` to fit the build allocation when compiling
FlashAttention, causal-conv1d, or Mamba. The default is two; Transformer Engine
retains its single-job bound.

The source checkout and recursive submodules must be clean. The script rejects
tracked changes and unexpected untracked files; Git-ignored build outputs remain
available for cache reuse.

Before publishing a wheel, install its exact bytes in the proposed runtime and
run its native forward and backward checks on the target GPU. Publish the source commits,
build environment, and checksums with the wheel assets. Adoption URLs and wheel
hashes belong in the root dependency manifest and lock.
