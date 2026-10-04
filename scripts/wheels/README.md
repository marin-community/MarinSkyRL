# CUDA 13.2 native wheels

`build_native.sh` builds pinned upstream sources against Torch 2.14.1+cu132 and
CUDA 13.2.2. `native-cu132.txt` pins the build environment, including NVCC, NVRTC,
CRT, NVVM and CCCL 13.2.86. Each resulting wheel needs runtime qualification
before adoption.

The recipes include Transformer Engine Torch 2.19, Megatron Core 0.19.2,
Megatron Bridge 0.6.2, FlashAttention 2.8.3.post1, causal-conv1d 1.7,
Mamba 2.3.2.post1 and fast-hadamard-transform 1.1 from the stable GitHub tag
`v1.1.0.post2`. Sources and recursive submodules
are fetched by exact commit. Upstream prebuilt Torch extensions are bypassed.

Core and Bridge carry explicit patches under `patches/`. Core retains Hero's
registered optimizer routes through its emerging optimizer factory and updates
dependency bounds for frontend 1.30, FLA 0.5.2 and OpenTelemetry 1.44. Bridge uses
Transformers through 5.18 and the paired FlashInfer 0.6.18.post1 packages. Both
Core uses the local version `+marin.torch2141.3`; Bridge uses
`+marin.torch2141.1`.

FlashAttention, causal-conv1d and Mamba force C++17 upstream. Their build patches
select C++20, which Torch 2.14 headers require. These wheels use the local version
`+marin.cu132torch2141.1`. Fast-hadamard-transform inherits Torch's compiler
standard without a patch.

The Mamba patch also incorporates the upstream TVM FFI upper-bound correction to
0.1.12 and selects the serving stack's TileLang 0.1.14. These are candidate compatibility changes; metadata
resolution alone does not qualify their APIs or kernels.

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

Use CPython 3.12.14 on Linux with git, a C++ compiler and uv. The multiarch Iris
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
bash scripts/wheels/build_native.sh flash-attn /tmp/build-flash-attn
bash scripts/wheels/build_native.sh causal-conv1d /tmp/build-causal-conv1d
bash scripts/wheels/build_native.sh mamba-ssm /tmp/build-mamba
bash scripts/wheels/build_native.sh fast-hadamard-transform /tmp/build-hadamard
```

Use a fresh build directory for a patched source. The applied patch stays in the
checkout for audit. The script requires a pristine checkout and submodules before
applying its declared patch; it rejects unexpected tracked or untracked changes.
Unpatched sources may retain Git-ignored build caches. Wheels appear in `dist/`,
with digests in `SHA256SUMS`.

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

`repair_cusparselt_wheel.py` accepts only the upstream wheel with SHA-256
`4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f`. It verifies
the native ELF architecture, corrects the internal platform tag, and regenerates
`RECORD`. All other members, including the native library and license, retain
their exact bytes. The output includes a provenance JSON file with input and
output hashes and hashes of every preserved member.

```bash
python3 scripts/wheels/repair_cusparselt_wheel.py \
  /tmp/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl \
  /tmp/repaired-wheels
```

Install and check the repaired wheel on an ARM host before adoption. Pin the
published wheel by URL and SHA-256. This packaging correction preserves the
vendor's native code and its compiler provenance.
