# Latest-first FA4 prototype

The baseline is MarinSkyRL `4d798b12`: Torch 2.13/CUDA 13.2, Megatron Core
0.18, Bridge 0.6, Transformer Engine 2.11, and FlashAttention 2.8.3. This
branch first tested Core 0.19.2, Bridge 0.6.2, Transformer Engine 2.19, and
FlashAttention 4 beta31. The retained lock uses beta29 after a full-causal
regression appeared in beta30 and persisted in beta31. FA2 remains the default;
select `--extra fa4` together with `--extra megatron` to install FA4.

The current [Marin FA2 wheel](https://github.com/marin-community/MarinSkyRL/releases/tag/native-cu132-fa283-20260920)
has SHA-256 `4b5086728757d81c8ef89f3008b0bcea72483cf18ed96e0e31b5293ed1f01bc7`
on x86_64 and `9cd0731dcc8fe780aeea28480ad80456d235fc0f9cb75d92c45d2b3c233eee5e`
on arm64/SM100. The retained
[upstream FA4 beta29 wheel](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta29)
has SHA-256 `85dd4b0e98ca38f5681a3214b2f9f230638030e3b868a50f3873336fc98d7537`.
The rejected [beta31 wheel](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta31)
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
Quack at 0.6.4 and CUTLASS DSL at 4.6.2, which FA4 beta29 permits.
The fixed vLLM wheel also pins Apache TVM FFI 0.1.11, whereas FA4 beta29
requires at least 0.1.12. The lock overrides that transitive pin to the
latest 0.1.14.post0; vLLM import and runtime behavior still need checking.

The `cache/` and `dist/` directories are ignored local scratch. No custom
repacked wheel is required or published. The experimental lock now also
selects the published SM100 FA2 and TE 2.19 arm64 wheels on GB200/Grace and
resolves the Megatron/FA4 extra there. Bridge declares Flash Linear Attention,
but Grug does not use its FLA backend and release 0.4.2 has no arm64 wheel, so
the lock limits that one transitive package to x86_64. The
[full arm64 frozen-lock import](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-arm-full-lock-import-9f284477)
passed with the initial beta31 lock, the fixed vLLM wheel, Bridge/Core, and
TE 2.19. Grug behavior and vLLM serving are separate accelerator gates.

The [GB200 arm64 TE build](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-te219-arm-build-11082662)
also succeeded from the same TE source and pinned native build environment.
Its [prerelease wheel](https://github.com/marin-community/MarinSkyRL/releases/tag/fa4-te219-cu132-20260920-694f3adf)
is 980,583 bytes, SHA-256
`185d4dd78a26623351d7ea22ac3a353d6c675bdee47f19b6d8839618473c82ac`.
The staged and independently downloaded release asset hashes match.
`preflight_gb200.sh <staged-wheel-URI> <SHA-256>` installs a disposable
Torch/TE/FA4 environment and checks one GB200 kernel forward/backward with
explicit backend logs. It is separate from the full arm64 lock and cannot
establish a Grug result.
Its first attempt installed TE and FA4 but import failed because `tvm_ffi` was
absent: the then-current project override and FA4 extra were x86_64-only. The
preflight resolves outside the project and explicitly pins the available
arm64 `apache-tvm-ffi==0.1.14.post0` wheel. The
[second GB200 backend probe](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-gb200-backend2-e99529c2)
selected FA4 beta31 in TE's log and completed finite forward/backward on
Torch 2.13.0+cu132. The separate project arm64 lock was added after this
probe; it still needs install/import and Grug validation.

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

The [full x86_64 frozen-lock import](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-full-lock-import-3ef87994)
passed with the fixed Marin vLLM wheel, Bridge 0.6.2, Core 0.19.2, TE 2.19,
and FA4 beta31. The first Grug worker import then found that Core 0.19.2
removed `dist_checkpointing.strategies.base`; the prototype deleted Marin's
obsolete global async queue setup because checkpoint saves are synchronous
(`async_sharded_save=False`). The next attempt reached its optimizer update and
found that Core's fused grad-norm path returns a device scalar; the strategy
now converts that boundary value to the Python float expected by the policy
metric reducer. These are prototype compatibility edits, not proof of
checkpoint or production training behavior.

The [one-H100 Grug toy test](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-pp1-toy3-6927610c)
passed an eager-HF reference comparison, Megatron forward/backward, and one
PPO optimizer update in each arm. Ray worker logs report FA2 2.8.3 and FA4
beta31 respectively. Valid-token FA4-versus-FA2 log-prob differences were
max/mean `0.01131/0.00222` before the update and `0.02715/0.00888` afterward.
The chosen attention-gate weight update differed by at most `4.77e-6`; the
layer-3 Q-projection update differed by at most `0.00189`. All compared values
were finite. The first-step wall times include initialization and JIT; they
must not be used as steady-state performance evidence.

The first two-H100 CP2 Grug attempt stopped before attention: Marin's Megatron
wrapper requires sample packing for CP, while the active Grug configuration
disables packing. The probe now enables packing only when CP2 is requested and
revalidates the configuration. A resulting CP2 success would establish the
packed experimental path, not current Grug production support.
That packed attempt reached TE but TE 2.19 rejected sliding-window attention
with its default `p2p` CP transport. The next probe sets MCore's documented
`cp_comm_type=all_gather`, which TE 2.19 explicitly supports with a sliding
window; this is a visible experimental transport choice, not a fallback.
The first `all_gather` run reached an FA2 optimizer step but stopped on the
probe's own positive-grad-norm assertion. Grug sets gradient clipping to zero,
and Core may report a zero norm in that mode. The probe now requires a finite,
nonnegative reported value plus an actual attention-weight change; zero alone
is neither accepted as gradient proof nor treated as a kernel failure.

The [next CP2 FA2 arm](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-cp2-health-fa2-5032510d)
reported `raw_grad_norm=nan`, with nonfinite values in all 8,192 sampled
layer-3 Q weights and all 128 sampled attention-gate weights after its one
update. This is not just a missing metric. The separately run
[CP2 FA4 arm](https://iris.oa.dev/#/job/%2Fromain%2Ffa4-grug-cp2-health-fa4-5032510d)
reported a finite norm and finite sampled weights, with an attention-gate
update. Both jobs selected their intended backend in TE logs. The FA4 result
does not license a CP migration: the matched new-cohort FA2 reference failed,
and the old-stack FA2 baseline remains to be checked.

`run_three_arm_h100.sh <world-size> <toy|snowball>` runs the two new-cohort
arms, then checks out baseline commit `4d798b12` in task-local scratch, installs
its frozen lock, and runs the same Grug probe against the old FA2 stack. It
compares valid-token log-probs and optimizer-induced changes in two attention
weights. Each environment remains separate; the fixed vLLM source is installed
in every arm. Use the output of this script only after the individual Grug
gates pass on the chosen topology.
