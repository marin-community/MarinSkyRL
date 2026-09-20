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
