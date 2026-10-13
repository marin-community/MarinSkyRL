<div align="center">

# SkyRL: A Modular Full-stack RL Library for LLMs


[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/) [![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL) [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI) [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI) [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN) [![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)



<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a> 
  </p>
</div>

</div>

# Overview of this fork

This is a fork of SkyRL maintained for the [Marin project](https://github.com/marin-community/marin) (`marin-community`), where it powers agentic RL training (SkyRL + Harbor). It was originally developed for the [OpenThoughts-Agent project](https://github.com/open-thoughts/OpenThoughts-Agent); that line of work now continues here under Marin.

MarinSkyRL is maintained as a hard snapshot; no upstream sync or merge-back is planned.

## MarinSkyRL packaging

The repository root builds one `marinskyrl` wheel containing the typed Iris launcher, `skyrl_train`, and
`skyrl_gym`. Its base dependency set is CPU-only and supports launcher inspection, validation, and dry runs:

```bash
uv sync --frozen
uv run --frozen marinskyrl --help
```

The [native Tinker reasoning reproduction](skyrl-train/ci/opd/tinker_repro/README.md#step-400-sft-and-one-step-opd-result)
records the step-400 SFT adapter, the one-step OPD gate, AIME 2024 evidence, and replay commands.

Python extras cannot replace a base CPU Torch wheel with a CUDA wheel, so `cpu` and `cuda` are mutually
exclusive wheel profiles. GPU training selects the `megatron` and `vllm` extras from the frozen root lock.
Iris installs this profile before launch. The `cpu` profile remains on Torch 2.11.0.

Iris training and checkpoint export select `IRIS_TASK_IMAGE` in
[the runtime environment](cloud/iris/runtime_environment.py). Update that value to a published
multi-architecture `iris-task` manifest digest after verifying its Python patch version on every
selected architecture. Validate frozen-profile installation, two-node H100 Ray admission, and
checkpoint export with the selected image before publishing the digest; GPU dependency versions
come from `uv.lock`.

## Task inputs and rollouts

Generated RL accepts prepared TaskSpecs in the `task_spec` dataset column.
SkyRL adds runtime and session settings, executes each task through RolloutEngine,
and builds whole-trajectory or per-step training batches directly from RolloutData.
The trainer does not convert raw dataset rows or Harbor directories.

See [dataset preparation](skyrl-train/docs/datasets/dataset-preparation.rst) for the input format.
See [Harbor preparation](skyrl-train/examples/terminal_bench/README.md) for package conversion.
See [task rollouts](skyrl-train/docs/tutorials/task_rollouts.rst) for session and machine settings.
The [GSM8K builder](skyrl-train/examples/gsm8k/gsm8k_dataset.py) prepares tasks for
the [training script](skyrl-train/examples/gsm8k/run_gsm8k.sh).

Verifier setup, opt-in configuration and offline comparison commands are documented in [Unified verification](docs/verifyit.md).

Offline DPO data formats and generated validation are documented in
[Preference training and task validation](docs/preference-training.md).
