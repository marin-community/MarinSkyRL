# Fully Async Training Example

Fully asynchronous (PipelineRL / AReal style) GRPO for Qwen2.5-1.5B-Instruct on GSM8K.

## Usage

```bash 
# prepare the dataset
uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/fully_async/async_run_gsm8k.sh
```

The fully async Gym runner calls the colocated inference router directly. It does not require
`generator.enable_http_endpoint`; enable the HTTP endpoint only when an external agent needs it.

For more details, refer to the documentation: https://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html

Especially, refer to the section on what knobs to tune: http://skyrl.readthedocs.io/en/latest/tutorials/fully_async.html#step-2-config-knobs-to-tune-for-fully-async-training
