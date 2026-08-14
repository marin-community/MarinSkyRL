# Out-of-band Hugging Face exports

Normal training writes sharded checkpoints and never assembles a Hugging Face
model on the live training ranks. At each configured export step, the trainer
writes `hf_export_request.json` beside the completed `global_step_N`
checkpoint. A separate Iris job consumes that request and performs conversion,
serialization, and optional Hugging Face Hub publication.

## Launcher storage policy

Both direct `cloud.iris.iris_backend` launches and typed Marin launches use the
same storage defaults. Resume checkpoints, raw Harbor traces, retained
trajectories, rendezvous state, and uploaded Ray session material use a
`tmp/ttl=14d` prefix. The launcher retains two resume checkpoints by default and
accepts an explicit limit of one or two. Ray object spill uses node-local
`/tmp/skyrl-ray-spill` storage unless remote spill is explicitly selected.

The terminal policy export and resolved launch configuration use a durable
`marin/users/<user>/skyrl/<job>/` prefix. Paths under durable `iris/` storage and
temporary paths without a lifecycle TTL fail validation before submission.

A successful synchronous launch submits and verifies the terminal export before
returning. A detached `--no-wait` launch returns after submission, so its caller
is responsible for terminal export handling.

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

Export jobs enter a dedicated checkpoint converter instead of the configured RL
entrypoint. The converter creates only the policy worker group at the geometry
recorded in the request. It does not initialize training or evaluation data,
rollout engines, a generator, tracking, reference or critic models, optimizers,
schedulers, profilers, or weight synchronization. Before model initialization,
it requires `trainer_state.pt` to record exactly the requested step.

## Trainer configuration

`trainer.hf_save_interval` follows `trainer.ckpt_interval` by default. An
explicit export interval must be a multiple of the checkpoint interval so every
request refers to a completed checkpoint.

Hub publication runs after conversion completes in the export job. Normal
training may record a Hub destination in the request, but it does not run the
upload callback on the training ranks.

## Failure boundary

The export job has its own timeout and exit status. FSDP2 and Megatron conversion
still require a distributed GPU gang at the saved policy geometry because their
checkpoint formats are sharded. The export-specific runtime profiles install the
selected FSDP2, DeepSpeed, or Megatron strategy without vLLM, Harbor, Daytona, or training telemetry.
Conversion does not hold unrelated training or rollout ranks while one rank
serializes or uploads the model.
