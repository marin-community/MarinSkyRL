# CUDA 13.2 native wheels

`build_native.sh` builds pinned upstream sources against Torch 2.14.1+cu132 and
CUDA 13.2.2. `native-cu132.txt` pins the build environment, including NVCC, NVRTC,
CRT, NVVM and CCCL 13.2.86. Each resulting wheel needs runtime qualification
before adoption.

The recipes include Transformer Engine Torch 2.20.2, Megatron Core 0.19.2,
Megatron Bridge 0.6.2, FlashAttention 2.8.3.post1 and
fast-hadamard-transform 1.1 from the stable GitHub tag
`v1.1.0.post2`. Sources and recursive submodules
are fetched by exact commit. Upstream prebuilt Torch extensions are bypassed.

Core and Bridge carry explicit patches under `patches/`. Core retains the
registered optimizer routes through its emerging optimizer factory and the
frontend 1.30 requirement exception. Its native FLA 0.4 and OpenTelemetry 1.43
bounds are preserved. Bridge preserves its native Transformers upper bound at
5.12.1 and carries the paired FlashInfer 0.6.18.post1 requirement exceptions.
Core uses the local version `+marin.torch2141.4`; Bridge uses
`+marin.torch2141.2`.

The one-time Core `.3` to `.4` metadata migration is complete. Its
[recipe](https://github.com/marin-community/MarinSkyRL/blob/cb2565d6207549199ecd4c31c343d0bb7af03a6c/scripts/wheels/restore_core_bounds.py)
remains at the qualified source, and the published wheels remain available.
New Core wheels build directly from the pinned source and current patches.

Bridge is pure Python. Its separate `bridge-build.txt` environment uses
setuptools 79.0.1 under the native `<80` requirement and CPU Torch 2.14.1;
CPython 3.12.13 builds this wheel without a CUDA toolkit or GPU. The builder does not change the managed Marin
runtime setuptools policy. Native trainer packages continue to use
`native-cu132.txt`. Build requirements and runtime requirements have separate
provenance.

The Speculators recipe retains the qualified `ecac1b1` implementation and its
Torch 2.14.1 bound exception, but restores the original Transformers `<5.17.0`
cap. Its native nightly packaging path avoids a build-host `file://` dependency
on hs-connectors. The pinned v0.8.0 tag and iteration 16 produce
`0.9.0a16.post1`; the post-release identifies the restored metadata.
The unchanged qualified hs-connectors source satisfies the native nightly
dependency range. This is another maintained pure Python artifact and version
patch, with no model or optimizer source changes. `speculators-build.txt` pins
its separate CPython 3.12.13 builder; runtime dependencies come from the root lock.

FlashAttention forces C++17 upstream. Its build patch
selects C++20, which Torch 2.14 headers require. Its wheels use the local version
`+marin.cu132torch2141.1`. Fast-hadamard-transform inherits Torch's compiler
standard without a patch.

Snowball and Hero use the maintained Torch ShortConv path. Their combined
runtime omits Mamba and causal-conv1d. The old build branches and their C++20
and Mamba dependency patches are retired; historical artifacts and recipes
remain available from their tagged releases.

Core checks the separately versioned `flash-attn-4` distribution before importing
its CuTe implementation. Its upstream import order loads the old CuTe code
bundled in the stable FA2 wheel even when no eligible FA4 distribution is
installed. That code cannot import with CUTLASS DSL 4.7. The patch preserves
Core's existing version gate and attention selection.

`build_native.sh` is a manual release tool. CI and runtime installation consume
published wheels by URL and SHA-256. Reproducibility here means pinned source,
dependencies and commands; separate builds can produce different archive bytes.
`BUILD_INFO` records the source commit, patch digest and applied diff, submodules,
compiler versions, recipe hashes and Torch ABI. `BUILD_REQUIREMENTS.txt` records
the installed build dependencies.

The existing CPU packaging suite checks retained workaround coverage against
the actual frozen GPU export on both architectures, the source commits in
`native_versions.json`, and the native and pure Python build requirements:

```bash
uv run --frozen --group dev pytest infra/tests/test_marinskyrl_packaging.py -q
```

A dependency or patched source outside that coverage fails even when the patch
still applies. Extend coverage only after testing removal and retaining new
necessity evidence beside the affected patch. This CPU check establishes input
coverage; accelerator behavior still needs the runtime gates below.

Use CPython 3.12.14 for native extensions and 3.12.13 for the pure Bridge and
Speculators builders on Linux with git and uv. Native builds also need a C++ compiler. The multiarch Iris
build image is
`ghcr.io/marin-community/iris-task@sha256:28a807a676b0b0ae155a80b1c6de75ae1b2932e25cef9bb8462b9ed7582c6990`.
Record the resolved architecture image and host tool versions with each build.
These are Linux wheels for the task environment; they do not claim manylinux
portability. CPU-only jobs need no GPU reservation.

Invoke through `bash` because Iris bundles do not retain executable modes:

```bash
bash scripts/wheels/build_native.sh transformer-engine-torch /tmp/build-te
bash scripts/wheels/build_native.sh megatron-core /tmp/build-core
bash scripts/wheels/build_native.sh megatron-bridge /tmp/build-bridge
bash scripts/wheels/build_native.sh speculators /tmp/build-speculators
bash scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn
bash scripts/wheels/build_native.sh fast-hadamard-transform /tmp/build-hadamard
```

Use a fresh build directory for a patched source. The applied patch stays in the
checkout for audit. The script requires a pristine checkout and submodules before
applying its declared patch; it rejects unexpected tracked or untracked changes.
Unpatched sources may retain Git-ignored build caches. Wheels appear in `dist/`,
with digests in `SHA256SUMS`.

FlashAttention uses the upstream
[`FLASH_ATTN_LOCAL_VERSION` option](https://github.com/Dao-AILab/flash-attention/blob/a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2/setup.py#L429)
to write its local version into the wheel filename, METADATA and dist-info
paths during the build. No archive rewrite is needed. The former retagger and
shared repacker are retired; their recipes remain at the qualified source.
The currently published wheel bytes and hashes remain unchanged.

FlashAttention selects SM90 on x86_64 and SM100 on aarch64. The other recipes set
`TORCH_CUDA_ARCH_LIST` to the corresponding target, though upstream build scripts
can add their own architectures. Inspect device code and run the actual selected
kernels on each target. A TE Torch rebuild replaces the host bridge; vendor
`libtransformer_engine.so` and runtime JIT paths have separate provenance.

Before adopting a wheel, install its exact bytes from the frozen profile and
check values and gradients against independent references on each architecture.
Megatron also needs context-parallel forward and backward with the FlashAttention
backend selected. Publish source, patches, build environment and checksums beside
the artifacts. Adoption URLs and hashes belong in the root manifest and lock.

The previous qualified Torch 2.13 artifacts remain documented in the
[original CUDA 13.2 release](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-torch2.13-20260910-6d12d7a0),
[FlashAttention release](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-fa283-20260920)
and [TE 2.19 release](https://github.com/marin-community/MarinSkyRL/releases/tag/fa4-te219-cu132-20260920-694f3adf).

## cuSPARSELt ARM platform tag

NVIDIA's cuSPARSELt 0.8.1 aarch64 wheel contains an aarch64 ELF library but declares
`py3-none-manylinux2014_sbsa` in `WHEEL`. The filename uses
`py3-none-manylinux2014_aarch64`. This mismatch makes `uv pip check` reject a clean
ARM installation of Torch 2.14.1+cu132, which requires cuSPARSELt 0.8.1.

The completed one-time repair used the exact upstream wheel with SHA-256
`4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f`.
Its [archived recipe](https://github.com/marin-community/MarinSkyRL/blob/cb2565d6207549199ecd4c31c343d0bb7af03a6c/scripts/wheels/repair_cusparselt_wheel.py)
checks the native ELF architecture and changes only `WHEEL` and `RECORD`.
The native library and license bytes remain identical. The published wheel and
its provenance remain available under `native-cu132-cusparselt081-armtag-20261004`.

The repair script is retired from the active build tools. Current builds reuse
the immutable repaired artifact by URL and SHA-256. This retains an artifact
exception; retiring its one-time tool does not remove that exception. Remove
the URL override when a qualified Torch release accepts a correctly tagged
native NVIDIA wheel. Newer cuSPARSELt 0.9/0.10 releases cannot replace 0.8.1
under Torch 2.14.1's exact native requirement. Verify native ARM installation
with `uv pip check` and the existing GPU/serving gates before removal.
