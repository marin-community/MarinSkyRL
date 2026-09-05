# Fully Async Training Example

Fully asynchronous (PipelineRL / AReal style) GRPO for Qwen2.5-1.5B-Instruct on GSM8K.
The non-agentic entrypoint uses the same direct SkyRL-Gym runner as synchronous
training, preserving engine token IDs and behavior logprobs for `behavior_clip`.
It uses the tokenizer's native chat template; an HTTP endpoint and a custom
retokenizing template are unnecessary. This example retains its FSDP2 backend.

## Usage

Run from `skyrl-train/` with the root project:

```bash
# prepare the dataset
uv run --project .. python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/fully_async/async_run_gsm8k.sh
```

See the maintained [fully async tutorial](../../docs/tutorials/fully_async.rst)
for batch geometry, producer capacity, staleness, and checkpoint behavior.
