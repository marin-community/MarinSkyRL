# Fully Async Training Example

Fully asynchronous (PipelineRL / AReal style) GRPO for Qwen2.5-1.5B-Instruct on GSM8K. Training and generation
run on separate GPUs, and generation may run up to `trainer.rollout_buffer.max_staleness_steps` (4 here) policy
steps ahead of training.

## Usage

```bash
# prepare the dataset
uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/fully_async/async_run_gsm8k.sh
```

The Gym worker pool calls the inference router directly. It does not require
`generator.enable_http_endpoint`; enable the HTTP endpoint only when an external agent needs it.

See `docs/tutorials/fully_async.rst` for the rollout buffer design and the settings to tune.
