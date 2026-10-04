# Start RL from a Megatron policy checkpoint

Set `trainer.policy.model.initial_checkpoint_path` to a completed checkpoint's
`global_step_<step>/policy` directory. Workers load only the model weights before
creating the optimizer, so its master parameters start from the loaded policy.
Optimizer moments, scheduler, trainer counters, and data-loader state start fresh.

The reference model's `initial_checkpoint_path` defaults to the policy setting.
This anchors KL to the warm-start policy. Set the reference field explicitly to
`null` to keep the reference at its configured Hugging Face weights.

```yaml
trainer:
  policy:
    model:
      initial_checkpoint_path: s3://bucket/checkpoints/global_step_64/policy
  resume_mode: latest
```

Use a new output checkpoint root for the RL run. Its later retries restore the
RL checkpoint, including optimizer and trainer state, through the existing resume
path. The initial checkpoint remains the reference anchor. The configured Hugging
Face model must match the checkpoint architecture and tokenizer. No Hugging Face
conversion is required; the Megatron distributed loader reads the policy shards.
