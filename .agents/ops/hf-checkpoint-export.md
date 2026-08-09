# Exporting training checkpoints to HF safetensors

`cloud/iris/export_hf_checkpoint.py` reuses the trainer's native export path. It resumes a selected
checkpoint with `max_steps` equal to the checkpoint step, reaches the train-end callback without an
optimization step, and writes the HF export to durable storage.

## Launch

```bash
python -m cloud.iris.export_hf_checkpoint \
  --ckpt_path <checkpoint-root> \
  --step <step> \
  --rl_config <config-used-for-training> \
  --model_path <base-model-used-for-training> \
  --cluster <cluster> \
  --num-nodes <training-node-count> \
  --gpus-per-node <training-gpus-per-node> \
  --job-name <unique-export-name>
```

Always inspect `--dry-run` output first. The configuration and parallel geometry must match the
checkpoint. Export execution uses a dedicated trainer lifecycle: it does not accept or load train
or evaluation data, does not construct a dataloader, and does not start the rollout generator. It
disables database registration. Hub publication runs only when the export request names a Hub
destination.

## Diagnose failures

- Any dataset load or dataset-size assertion is an export-contract regression; export jobs do not
  depend on training data.
- An empty node-local destination means `export_path` was not durable and visible to the callback.
- Teardown process noise often follows the real error; inspect a bounded tail large enough to cover
  the first causal traceback.
- A successful controller exit is insufficient. Verify the object-store destination.

## Verify output

Check `model.safetensors.index.json`, every referenced shard, `config.json`, and tokenizer files.
The index total size must agree with the actual shards, every mapped tensor must resolve, and shard
names/counts must be internally consistent. A partial shard set may indicate an active writer; use
object timestamps and job state before declaring failure.

Object-store access details live in `iris-operator-scripts.md`. Publication and registration rules
live in `checkpoint-consolidation.md` and the campaign record.
