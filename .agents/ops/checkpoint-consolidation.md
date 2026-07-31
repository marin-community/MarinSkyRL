# Checkpoint consolidation and publication

Use the checkpoint layout and resolved trainer strategy to choose the export route.

## Identify the format

| strategy | layout under `global_step_N/policy/` | route |
|---|---|---|
| `fsdp` / `fsdp2` | `model_world_size_*_rank_*.pt` | local consolidation script |
| `megatron` | distributed-checkpoint shards plus metadata | Iris export job |

Do not hand-remap Megatron tensors. Its HF conversion requires the Megatron runtime, a process
group, and the original parallel geometry.

## FSDP/FSDP2

Inspect before writing:

```bash
cd skyrl-train
uv run python scripts/consolidate_sharded_checkpoint.py \
  --src <checkpoint>/global_step_<step>/policy --inspect
```

Then consolidate:

```bash
uv run python scripts/consolidate_sharded_checkpoint.py \
  --src <checkpoint>/global_step_<step>/policy \
  --dst <staging>/model \
  --aux-from <directory-or-hf-repo>
```

Add `--defuse-moe` only when inspection reports a supported fused grouped-MoE layout. The script
must fail rather than produce partial output when ranks, parameter sets, placements, or tensor sizes
disagree.

## Megatron

Follow `hf-checkpoint-export.md` for the canonical command, required inputs, dry-run gate,
validation, and failure signatures.

## Validate and stage

Before publication:

1. Apply the output checks in `hf-checkpoint-export.md`.
2. For a locally consolidated FSDP model, also compare exported parameter count with the inspection
   result.
3. Load the staged model metadata, and the model itself when feasible.
4. Copy model files to the staging root; place resolved configuration and logs under stable names.
5. Redact the complete staging tree before upload:

   ```bash
   python -m infra.rl_cleanup.secret_redaction <staging>
   ```

   Retain the emitted findings with the publication record. Redaction is required for trainer logs
   and applies recursively to all UTF-8 staging files.

## Publish safely

Use the campaign record for namespace, visibility, naming, and registration policy.

```bash
hf upload <namespace>/<repo> <staging-directory> --repo-type model
```

- Use additive upload behavior. Do not use a client mode that deletes remote files absent locally.
- Long uploads belong in a resumable terminal session.
- Derive size and lineage from the artifact and resolved configuration, not names.
- Download the published repository into a fresh directory, then run both checks before reclaiming
  any source or staging storage:

  ```bash
  python -m infra.rl_cleanup.secret_redaction <fresh-download> --check
  python -m infra.rl_cleanup.publication_checks <fresh-download>
  ```

  The model check requires weights, a nonempty `training_logs/` directory, and a `Training Traces`
  model-card link to the companion Hugging Face dataset.
