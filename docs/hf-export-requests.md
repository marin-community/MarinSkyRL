# Out-of-band Hugging Face exports

Normal training writes sharded checkpoints and never assembles a Hugging Face
model on the live training ranks. At each configured export step, the trainer
writes `hf_export_request.json` beside the completed `global_step_N`
checkpoint. A separate Iris job consumes that request and performs conversion,
serialization, and optional Hugging Face Hub publication.

## Request lifecycle

The request is written only after the source checkpoint completes. It records
the immutable checkpoint path, export destination, model and parallel geometry,
Hub settings, attempt count, timeout, and last exit code. A task-local model path
must include the object-store source and immutable identity needed to materialize
the same model in the export gang. A Hugging Face repository ID is independently
resolvable and does not require a source URI.

The export job marks the request `in_progress` before submission. A successful
Iris job marks it `complete`; a failed job returns it to `pending`, so the same
request can be retried. Pending, in-progress, and malformed requests protect
their source checkpoints from retention cleanup. Export destinations may
contain partial files after failure and must not be treated as complete until
the request says `complete`.

Request discovery and scheduling remain orchestration responsibilities. Consume
a specific request with:

```bash
uv run --frozen python -m cloud.iris.export_hf_checkpoint \
  --request s3://bucket/run/checkpoints/global_step_N \
  --rl_config cloud/iris/configs/example.yaml \
  --cluster <cluster> \
  --timeout 7200
```

The request owns checkpoint, model, geometry, export destination, and Hub
settings. The command owns operational placement, priority, and timeout.
Conflicting request-owned command-line options fail before submission.

Export jobs do not accept or load training or evaluation data. They also skip
generator startup. After loading the checkpoint, the trainer requires its
recorded global step to equal the request step before it runs finalization.

## Trainer configuration

`trainer.hf_save_interval` follows `trainer.ckpt_interval` by default. An
explicit export interval must be a multiple of the checkpoint interval so every
request refers to a completed checkpoint. `trainer.hf_export_execution` is
reserved for the export job and must remain false during normal training.

Hub publication runs after conversion completes in the export job. Normal
training may record a Hub destination in the request, but it does not run the
upload callback on the training ranks.

## Failure boundary

The export job has its own timeout and exit status. Conversion no longer holds
non-exporting training ranks at a trailing process-group barrier while one rank
serializes or uploads the model. FSDP2, DeepSpeed, and Megatron retain the
collectives required to gather weights; they do not add a barrier after the
single-writer serialization phase.
