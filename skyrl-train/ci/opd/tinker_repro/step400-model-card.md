---
base_model: Qwen/Qwen3.5-9B-Base
library_name: peft
tags:
  - lora
  - qwen3.5
  - experimental
---

# Native Tinker-style SFT, step 400

This repository preserves the fused-QKV LoRA adapter from step 400 of a
3,000-step OpenThoughts3 SFT run on `Qwen/Qwen3.5-9B-Base`. It is an
intermediate checkpoint, not the final Tinker reproduction. Load it with base
revision `68c46c4b3498877f3ef123c856ecfde50c39f404`.

The rank-128 Axolotl SFT used the seed-0, 384,000-row OpenThoughts3 stream.
The conversion from split Q/K/V factors to fused QKV was performed by the
Marin Axolotl fork at revision `d5ae94ae7446d3f3fc4ebc8d97fd9d00319f9811`.
The original Axolotl SFT and Tinker-hosted optimizer are not identical.

The repository root contains the fused adapter for inference and OPD loading.
`sft-training-checkpoint-400/` contains the original 20-file Axolotl
checkpoint, including its split-QKV adapter, optimizer, scheduler, RNG,
tokenizer, trainer state, and `checkpoint-commit.json`. This folder is needed
to resume SFT rather than merely load the step-400 weights. The original is
also committed at
`s3://marin-us-east-02a/iris/cw-rno2a/experiments/tinker-native-repro/20260916t/sft-full/peft/checkpoint-400/`.

| File | SHA-256 |
| --- | --- |
| `adapter_config.json` | `ab604790f400d65bc8f53a221de417c5c559d68031184a82f49e380d5362b131` |
| `adapter_model.safetensors` | `50a4510256c71a1cd0f49ba1866903660a38ce6e06a8d8d0cb7daaa3a8d945b0` |

The original split-QKV adapter in `sft-training-checkpoint-400/` has SHA-256
`684f168d3ed1ff4a186ee1ebc1c9e92a22ef382f3b537013f397f88a976e1352`;
its `optimizer.pt` has SHA-256
`376e4aec0b2555a338f757d2b272b54dcf72f1e88075ea222d6f18a9bd18a445`.

One-sample AIME 2024 evaluation of this adapter scored 19/30, with no
truncations. One native MarinSkyRL OPD update from it, using 512 DeepMath
prompts and four student rollouts per prompt, produced a separately evaluated
checkpoint scoring 26/30 raw and 26/28 among completed responses. Two
responses truncated; that evaluation is marked non-comparable in its manifest.
Stochastic reruns need not reproduce the same single-sample score.

The [MarinSkyRL reproduction README](https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/ci/opd/tinker_repro/README.md#step-400-sft-and-one-step-opd-result)
records the replay command, model and data revisions, scoring protocol, and
durable experiment artifacts.
