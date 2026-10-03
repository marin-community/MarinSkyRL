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

Transformer Engine targets official 2.19 source
`5e52befd5262c06289106338c308079d6adb391f` on both x86_64 and aarch64.
The [published wheels](https://github.com/marin-community/MarinSkyRL/releases/tag/fa4-te219-cu132-20260920-694f3adf)
use CPython 3.12's `cp312-cp312` ABI and Torch's C++11 ABI. Their hashes are:

| Architecture | SHA-256 |
| --- | --- |
| x86_64 | `1eb84026d9617aed8c656d91877d19ff70647697d6a2993de90063e6a6c37d56` |
| aarch64 | `185d4dd78a26623351d7ea22ac3a353d6c675bdee47f19b6d8839618473c82ac` |

The [H100 build](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-te219-build2-543a0ba8)
used [recipe revision 543a0ba8](https://github.com/marin-community/MarinSkyRL/blob/543a0ba8012604104f99c3113632aa8e6c3d9ad0/scripts/wheels/build_native.sh).
The [GB200 build](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-te219-arm-build-11082662)
used [revision 11082662](https://github.com/marin-community/MarinSkyRL/blob/11082662/scripts/wheels/build_native.sh).
Those revisions called the same TE recipe `transformer-engine-torch-2.19`.
The maintained `transformer-engine-torch` target uses the same source, build
requirements, environment and compilation command. Architecture checks and
provenance output surround that command. The unchanged build requirements have
SHA-256 `48282703aecb3a4c1dcfc46f0d4530ce1a9d1da9918061fa54a5c02d7ad91e13`.

Every source is fetched by exact commit. The pinned 2.8.3 FlashAttention wheel
was built with an unused package excluded; its
[release record](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-fa283-20260920)
preserves that provenance. This script now builds unmodified upstream source,
so a new wheel needs its own qualification and release.

Use CPython 3.12.14 on Linux with git, a C++ compiler, and uv. The
qualified FlashAttention 2.8.3 build used the Iris task image
`ghcr.io/marin-community/iris-task@sha256:ecdb2f7f90f8760a7e74c49b49b67d7ecf44557298860411c148c186706067f2`,
GCC/G++ `14.2.0-19`, glibc `2.41-12+deb13u3`, git `1:2.47.3-0+deb13u1`, and
uv `0.10.3`. Install the compiler and git inside the build container. These are
Linux wheels for that task environment; they do not claim manylinux portability.

For TE 2.19, pin the multi-architecture task image
`ghcr.io/marin-community/iris-task@sha256:28a807a676b0b0ae155a80b1c6de75ae1b2932e25cef9bb8462b9ed7582c6990`.
The maintained recipe built successfully in CPU-only x86_64 and aarch64 tasks
with CPython `3.12.14`, uv `0.10.3`, GCC/G++ `14.2.0-19` and glibc
`2.41-12+deb13u4`. The pinned requirements supply NVCC `13.2.78`, Torch
`2.13.0+cu132` and setuptools `80.10.2`. Each task requested two CPU cores,
16 GiB memory and 64 GiB disk. Preserve the image digest with the build records;
select the architecture using the live cluster topology rather than requesting
a GPU for compilation.

```bash
bash scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn
bash scripts/wheels/build_native.sh causal-conv1d /tmp/build-causal-conv1d
bash scripts/wheels/build_native.sh mamba-ssm /tmp/build-mamba
# On an x86_64 build task:
bash scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te-x86_64
# On an aarch64 build task:
bash scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te-aarch64
```

Each directory retains its environment and source build cache. Wheels are written
to `dist/`, with their SHA-256 digests in `SHA256SUMS`. `BUILD_INFO` records source
and recipe hashes, submodules, architecture, Python, uv, compiler, glibc, NVCC
and Torch ABI. `BUILD_REQUIREMENTS.txt` records the installed build environment.
FlashAttention targets SM90; the other projects retain their upstream
architecture choices. Only Transformer Engine is enabled for aarch64 here.

Launch native builds as CPU-only tasks on the required architecture; a build
does not need an allocated GPU. Preserve the task image digest and the package
versions above with each build. Iris bundles do not retain executable modes,
so invoke the script through `bash`.

The source checkout and recursive submodules must be clean. The script rejects
tracked changes and unexpected untracked files; Git-ignored build outputs remain
available for cache reuse.

Before publishing a wheel, install its exact bytes in the proposed runtime and
run its native forward and backward checks on each target architecture. Megatron also needs a
context-parallel forward and backward with the FlashAttention backend selected;
native attention checks alone cannot detect a Transformer Engine version cap.
Publish the source commits, build environment, and checksums with the wheel
assets. Adoption URLs and wheel hashes belong in the root dependency manifest
and lock.
