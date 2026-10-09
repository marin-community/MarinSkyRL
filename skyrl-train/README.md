# SkyRL-Train

MarinSkyRL trains language models with GRPO and related policy-gradient objectives.
It uses Megatron for GPU training and vLLM for inference.
The CPU profile supports launcher tests and small training tests.

[TaskCompendium](https://github.com/marin-community/marin/tree/main/lib/taskcompendium) defines task inputs and private grading rules.
The shared rollout engine calls the model and retains exact tokens from model responses and observations.
Each task session executes task actions and grades one attempt.
MarinSkyRL controls concurrency and retries, grades groups of attempts, and converts rollouts into training batches.
See the [rollout guide](docs/tutorials/task_rollouts.rst) and [session tutorial](docs/tutorials/new_env.rst).

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).
The GPU profile installs the CUDA 13.2 toolkit on GPU workers. Frozen dependencies select Ray 2.51.1.

```bash
git clone https://github.com/marin-community/MarinSkyRL
cd MarinSkyRL
uv sync --frozen --extra megatron --extra vllm
```

All packages share the root environment and lock.
For CPU tests, select `--extra cpu` instead of the GPU extras.
The [Iris runtime configuration](../cloud/iris/runtime_environment.py) selects and installs the frozen profile before launch.
See [installation](docs/getting-started/installation.rst) for worker and cluster setup.

## Run the GSM8K example

From `skyrl-train`, prepare the dataset and run the four-GPU example:

```bash
uv run --frozen python examples/gsm8k/gsm8k_dataset.py
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
LOGGER=console bash examples/gsm8k/run_gsm8k.sh
```

The Ray hook carries the `uv` environment into worker processes.
The script accepts configuration overrides as command arguments, such as `trainer.epochs=1`.
`NUM_GPUS`, `DATA_DIR`, and `LOGGER` set environment-variable defaults.
Set `WANDB_API_KEY` and use `LOGGER=wandb` for Weights & Biases logging.

## Documentation

- [Configuration](docs/configuration/config.rst)
- [Task rollouts](docs/tutorials/task_rollouts.rst)
- [New task sessions](docs/tutorials/new_env.rst)
- [SWE tasks](examples/mini_swe_agent/README.md)

## Source and citation

This fork derives from [SkyRL](https://github.com/NovaSky-AI/SkyRL).
The upstream project was developed at Berkeley Sky Computing Lab with Anyscale and other contributors.
See the [repository README](../README.md#acknowledgement) for acknowledgements.

```bibtex
@misc{griggs2025skrylv01,
      title={Evolving SkyRL into a Highly-Modular RL Framework},
      author={Tyler Griggs and Sumanth Hegde and Eric Tang and Shu Liu and Shiyi Cao and Dacheng Li and Charlie Ruan and Philipp Moritz and Kourosh Hakhamaneshi and Richard Liaw and Akshay Malik and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
      note={Notion Blog}
}
```
