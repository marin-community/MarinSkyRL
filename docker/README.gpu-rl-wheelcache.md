# gpu-rl image — wheel cache for fast rebuilds

The `gpu-rl` image needs two sets of native CUDA files built against torch
2.11 / CUDA 12.8 / cp312 / x86_64:

- the **vLLM fork native donor** (`marin-community/vllm` @ `4b555913`);
- **flash-attn 2.8.3** — its `flash_attn_2_cuda` extension.

The final vLLM distribution does not use the donor's Python package. It installs
the exact `VLLM_FORK_COMMIT` Git source and asks vLLM's supported precompiled
build path to extract only the native and Rust artifacts from the donor wheel.
This lets a dependency-only or Python-only vLLM change reuse the validated
native build without a multi-hour `nvcc` compile.

For genuine native changes, `Dockerfile.gpu-rl` retains a **`wheel-builder`**
stage that compiles both wheels. The **`rl`** stage validates the selected
native wheel manifest before installing flash-attn and the exact vLLM source.

## Build host / mechanism

The production mechanism is `docker/build_gpu_rl_kaniko.sh` on Iris; a local
x86 builder can use **`docker buildx build … --push`** directly. The production
`gpu-rl` image is **single-platform `linux/amd64`** (verified on the live digest)
— CoreWeave H100 + the x86 CUDA build are amd64-only. So everything below
targets `linux/amd64`.

> This must run on an **x86_64 / linux/amd64** host (real x86 build host, GPU
> build pod, or x86 CI runner). The nvcc compiles need `MAX_JOBS=8 * ~5GB/job ≈
> 40GB RAM`. On the arm64 dev Mac the amd64 pass runs under QEMU emulation
> (impractically slow + RAM-bound on Docker Desktop) — do NOT build there.

## The wheel cache

| | |
|---|---|
| **Cache location** | `docker/wheelhouse/` on the build host (generated wheels and `MANIFEST` are gitignored; only `.keep` is committed). Holds `vllm-*.whl`, `flash_attn-*.whl`, `MANIFEST`. |
| **Python source** | `VLLM_FORK_COMMIT` identifies the exact Git source and emitted dependency metadata installed in the image. |
| **Prebuilt native cache key** | the tuple `{ VLLM_NATIVE_DONOR_COMMIT, FLASH_ATTN_VERSION, torch 2.11.0, CUDA 12.8, cp312, x86_64, TORCH_CUDA_ARCH_LIST "8.0;9.0" }`. A source-only change does not invalidate it; a native input change does. |
| **Not cached** | torchtitan (pure-python, trivial — installed fresh each build) and all pip-resolved deps. |

## Commands

### One-time / on a pin change — build the wheels

```bash
source ~/secrets.env          # ghcr auth, if pushing later
./docker/build_wheels.sh      # -> docker/wheelhouse/{vllm,flash_attn}-*.whl + MANIFEST
```

Re-run this **only** when a pin/ABI in the cache key changes.

### Every rebuild — build + push from a populated wheelhouse (NO nvcc)

Run `build_wheels.sh` first, or otherwise stage a matching manifest and exactly
one wheel for each package in `docker/wheelhouse/`.

```bash
source ~/secrets.env
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <user> --password-stdin
NATIVE_DONOR_COMMIT=$(
  sed -n 's/^VLLM_FORK_COMMIT=//p' docker/wheelhouse/MANIFEST
)
docker buildx build -f docker/Dockerfile.gpu-rl \
  --platform linux/amd64 --target rl \
  --build-arg WHEEL_SOURCE=prebuilt-wheelhouse \
  --build-arg VLLM_NATIVE_DONOR_COMMIT="$NATIVE_DONOR_COMMIT" \
  -t ghcr.io/<owner>/<package>:gpu-rl-<gitsha> --push .
# capture the pushed @sha256 digest from the immutable :gpu-rl-<gitsha> tag:
docker buildx imagetools inspect ghcr.io/<owner>/<package>:gpu-rl-<gitsha>
```

### Iris rebuild from the validated wheel artifact

`docker/build_gpu_rl_kaniko.sh` must run only inside its disposable Ubuntu Iris
task. It installs system packages and overlays the Kaniko root filesystem onto
`/`; never invoke it directly on a workstation or shared host. Follow the
[Iris GPU-RL build guide](../.agents/skills/build-gpu-rl-image-iris/SKILL.md),
which submits the script inside the task container.

Use the artifact URI and SHA-256 recorded in the
[Iris GPU-RL build guide](../.agents/skills/build-gpu-rl-image-iris/SKILL.md).
That guide is the single operational source for the current validated artifact.

The archive must contain `wheels/MANIFEST`, one `vllm-*.whl`, and one
`flash_attn-*.whl`. The launcher verifies the archive digest and exact
native-donor ABI manifest, then the Dockerfile installs the exact Git source
with `VLLM_PRECOMPILED_WHEEL_LOCATION` pointing at that local wheel. A fetch,
hash, manifest, or precompiled-install failure is fatal; it never falls through
to a source compile or upstream nightly wheel. Iris injects the `FSSPEC_S3`
configuration and S3 credentials used to read this URI. A new manifest requires
a new immutable artifact and checksum; do not overwrite the existing object.
The fetch path compares the existing archive's legacy `VLLM_FORK_COMMIT`
manifest field with the Dockerfile's `VLLM_NATIVE_DONOR_COMMIT`.

### Explicit source rebuild (slow)

```bash
docker buildx build -f docker/Dockerfile.gpu-rl \
  --platform linux/amd64 --target rl \
  --build-arg WHEEL_SOURCE=wheel-builder \
  -t ghcr.io/<owner>/<package>:gpu-rl --push .
```

`WHEEL_SOURCE=wheel-builder` compiles `VLLM_FORK_COMMIT` and flash-attn afresh.
The resulting vLLM wheel becomes the native donor for that same source commit.
This is equivalent to the fast path but pays the `nvcc` cost.

## Shared release only — bump the launcher digest

Scratch builds must pass their immutable digest through `--docker_image` and
leave `DEFAULT_RL_DOCKER_IMAGE` unchanged. For an approved shared release, set
the default in `cloud/iris/launch_rl_iris.py` to the new digest (from the
`:gpu-rl-<gitsha>` tag, never the floating `:gpu-rl`) and update the provenance
comment.

## What proves the build is good

The `rl` stage asserts at build time:

- the staged wheel manifest matches the Dockerfile's native-donor and ABI pins;
- the vLLM checkout matches `VLLM_FORK_COMMIT`;
- vLLM's precompiled installer consumed the local donor wheel;
- the installed vLLM metadata contains the expected CUTLASS 4.5.x range and
  CUDA 12 Humming Kernels extra;
- `import flash_attn, flash_attn_2_cuda` — the CUDA extension EXISTS (from the wheel).
- `import torch, vllm, skyrl_train, flash_attn, flash_attn_2_cuda`.
- `from torchtitan.distributed.expert_parallel import ExpertParallel` — the
  **EP>1 MoE unblock**; if this line prints `... import OK`, `apply_ep` will
  resolve `ExpertParallel` and the CoreWeave EP=8 RL jobs can launch.
- the locked S3 client imports and constructs a client;
- optional Megatron imports resolve when `INSTALL_MEGATRON=1`;
- the pinned Harbor commit installs and imports.
