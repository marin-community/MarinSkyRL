# Consolidating a sharded training checkpoint into a publishable model

Facts and commands. Skills point here.

## What the two artifacts are

| written by | layout | publishable |
|---|---|---|
| `FSDPStrategy.save_checkpoint` (every `ckpt_interval`) | `model_world_size_{W}_rank_{R}.pt` — FSDP **sharded** DTensors, one file per rank, plus `optim_*` and `extra_state_*` | no |
| `FSDPStrategy.save_hf_model` (gated on `hf_save_interval`) | HF safetensors + `config.json` + tokenizer | yes |

A run can finish its full step budget with only the first. That is not a failure to detect and
report — it is a conversion to perform.

## Tool

`skyrl-train/scripts/consolidate_sharded_checkpoint.py` — reassembles the per-rank shards into
full tensors and writes an HF safetensors directory. CPU only, no process group, no GPUs.

**Always `--inspect` first.** It reports world size, parameter count, dtypes and every distinct
shard placement, and writes nothing. Reassembly depends on the mesh the run used, so read the
report before converting.

```bash
cd skyrl-train
uv run python scripts/consolidate_sharded_checkpoint.py --src <ckpt>/global_step_N --inspect
```

Then convert:

```bash
uv run python scripts/consolidate_sharded_checkpoint.py \
  --src <ckpt>/global_step_N \
  --dst <staging>/model \
  --aux-from <dir or HF repo id with config.json + tokenizer>
```

Add `--defuse-moe` when `--inspect` reports fused grouped-MoE weights (keys containing
`.moe.experts.` or `.experts.w1`). Without it the output loads in neither HF nor vLLM, because
`config.json` says `qwen3_moe` while the tensors are in the grouped layout. The de-fusion itself is
delegated to `convert_fused_moe_to_hf.py`, which stays in lockstep with the trainer's own remap.

## Failure behaviour

The script aborts rather than writing a partial model when:

- a rank file is missing (`world_size=W` but fewer than `W` ranks present),
- ranks disagree on the parameter set,
- a parameter is sharded across more than one dimension,
- a reassembled tensor does not match the size the DTensor declares.

A silently wrong set of weights costs more than a failed conversion, so none of these are
downgraded to warnings.

## Verify before publishing

`config.json` and the tokenizer files must be present in `--dst`; the script warns when
`config.json` is missing but still writes the weights. Confirm the parameter count in the final
report matches the `--inspect` count, and load the directory once before uploading.
