# Latest-first FA4 prototype

The baseline is MarinSkyRL `4d798b12`: Torch 2.13/CUDA 13.2, Megatron Core
0.18, Bridge 0.6, Transformer Engine 2.11, and FlashAttention 2.8.3. This
branch tests Core 0.19.2, Bridge 0.6.2, Transformer Engine 2.19, and
FlashAttention 4 beta31. FA2 remains the default; select `--extra fa4` together
with `--extra megatron` to install the FA4 distribution.

The current [Marin FA2 wheel](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-fa283-20260920)
has SHA-256 `4b5086728757d81c8ef89f3008b0bcea72483cf18ed96e0e31b5293ed1f01bc7`
on x86_64. The [upstream FA4 beta31 wheel](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta31)
has SHA-256 `6eda5890b29e90fc46e19a47b4018effae7c042f75ee8aaaeebd3f56ccd82edf`.
An archive audit found zero `flash_attn/cute` files in Marin's FA2 wheel, 52
in stock FA4, and no overlapping file paths. A no-dependency `uv pip install`
of both wheels into a temporary target succeeded and exposed both distribution
versions. Unlike the upstream SkyRL FA2 wheel, Marin's FA2 wheel therefore does
not need a combined-wheel repack. This is packaging evidence, not a GPU or
backend-selection result.

The newest Megatron lock needs Hydra 1.3.4. Bridge 0.6.2 imports Megatron
Core's dev extra, which pins cuDNN Frontend 1.26, while Transformer Engine
2.19 requires at least 1.28. The lock uses a narrow 1.29 override; its runtime
compatibility remains an accelerator gate. Marin's fixed vLLM package keeps
Quack at 0.6.4 and CUTLASS DSL at 4.6.2, which FA4 beta31 permits.
The fixed vLLM wheel also pins Apache TVM FFI 0.1.11, whereas FA4 beta31
requires at least 0.1.12. The lock overrides that transitive pin to the
latest 0.1.14.post0; vLLM import and runtime behavior still need checking.

The `cache/` and `dist/` directories are ignored local scratch. No custom
repacked wheel is required or published. The current `megatron` extra is
x86_64-only, so GB200/Grace needs an explicit arm64 closure before it can
support a Grug result. No CPU packaging check proves that GPU gate.

For the native TE 2.19 build, use
`bash scripts/wheels/build_native.sh transformer-engine-torch-2.19 <build-dir>`
on an H100 Iris task. Iris bundles do not preserve executable script mode, so
the `bash` prefix matters. The task image provides CPython 3.12.14, git, and a C++
compiler. Its bare Python lacks `boto3`; run the uploader through
`uv run --no-project --with boto3==1.42.97 python experiments/fa4/upload_candidate.py`
so it can use the cluster-injected S3 credentials. Upload the resulting wheel
to an immutable key in
`s3://marin-us-east-02a/iris/fa4-experiment/` and preserve the Iris job ID,
source commit, build script, pinned environment, and SHA-256. This staging key
is not a runtime wheel source; the lock references the published, verified
candidate artifact. `download_candidate.py` fetches the staged object
inside an Iris task and refuses a missing or mismatched SHA-256. Like the
uploader, run it through `uv run --no-project --with boto3==1.42.97 python`.

The x86_64 candidate is now published as a
[prerelease wheel](https://github.com/marin-community/MarinSkyRL/releases/tag/fa4-te219-cu132-20260920-694f3adf).
`uv.lock` records SHA-256
`1eb84026d9617aed8c656d91877d19ff70647697d6a2993de90063e6a6c37d56`;
the downloaded release asset matched the staged wheel. The
[H100 build](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-te219-build2-543a0ba8)
used official TE source `5e52befd5262c06289106338c308079d6adb391f`
and the pinned Torch 2.13/CUDA 13.2 build environment. This is a candidate,
not a production-qualified dependency.

The [H100 task preflight](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b31-h100-preflight-f6fed696)
completed on `cw-rno2a` with Python 3.12.14, uv 0.10.3, git 2.47.3, GCC
14.2, and driver 595.71.05. The [storage preflight](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-b31-s3-preflight2-f6fed696)
confirmed that an isolated `boto3==1.42.97` environment reaches the Marin
bucket from an Iris task. Neither preflight installed Transformer Engine.

`probe_attention.py` measures a fixed Transformer Engine causal/sliding-window
GQA forward and backward pass, records ten per-process samples and peak CUDA
allocation, and writes pointwise outputs and Q/K/V gradients. Run each arm in
its own process with `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 NVTE_FUSED_ATTN=0
NVTE_FLASH_ATTN=1` and explicitly set `NVTE_FLASH_ATTN_V4=0` for FA2 or `1`
for FA4. Inspect the Transformer Engine `Selected backend` log; a package
version alone is not backend proof. Use `compare_attention.py` on output files
from matched shapes and hardware to report max/mean/RMS differences. This
microprobe is not a substitute for the Grug CP2 optimizer or RL timing gates.
`preflight_h100.sh <staged-wheel-URI> <SHA-256>` installs a disposable, binary-only
Torch/TE/FA2/FA4 environment and runs the two 32-token kernel probes with a
ten-minute bound on each arm. It is an import/backend gate, not a lockfile or
performance conclusion.

The [H100 backend preflight](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-h100-backend2-694f3adf)
passed with TE 2.19.0, FA2 2.8.3, FA4 beta31, Torch 2.13.0+cu132, and
CUTLASS DSL 4.6.2. TE logged selection of FA2 and FA4 respectively. On its
32-token causal/sliding-window BF16 shape, outputs and Q gradients matched
exactly; max absolute K/V gradient difference was `3.8147e-6`, and every
recorded tensor was finite. Two samples per arm are too few for a performance
claim. The first attempt used the same kernels successfully but failed after
them because the pointwise comparison loader rejected a saved TorchVersion
object; commit `694f3adf` corrected that recorder error.
