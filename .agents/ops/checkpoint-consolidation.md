# Consolidating a sharded training checkpoint into a publishable model

Facts and commands. Skills point here.

## Identify the format FIRST — there are two, and they need different tools

`trainer.strategy` decides which one a run wrote. Check it before anything else.

| strategy | checkpoint layout | consolidation route |
|---|---|---|
| `fsdp` / `fsdp2` | `model_world_size_{W}_rank_{R}.pt` — per-rank DTensor shards | `consolidate_sharded_checkpoint.py` (below) |
| `megatron` | `__{N}_0.distcp` shards + `.metadata` + `metadata.json` saying `{"sharded_backend": "torch_dist"}` | **not yet available offline** — see below |

Both layouts sit under `global_step_N/policy/`, not directly under `global_step_N`.

The exported model, when it exists, is written by `save_hf_model` (gated on `hf_save_interval`)
as HF safetensors plus `config.json` and tokenizer files. A run can finish its full step budget
with only checkpoints and no export.

### megatron runs have no offline route yet

`MegatronStrategy.save_hf_model` converts through `bridge.save_hf_weights` (mbridge), which needs
the Megatron runtime and a live process group — it cannot run on a laptop. A megatron checkpoint
also needs **two** remaps, not one: Megatron-native layer-stacked keys to HF
(`decoder.layers.self_attention.linear_qkv.weight` is `(48, 5120, 2048)`), and the grouped expert
tensors (`decoder.layers.mlp.experts.experts.linear_fc1.weight`, `(48, 128, 1536, 2048)`) to
per-expert HF layout. `--defuse-moe` handles only the second, and only for `.moe.experts.` /
`.experts.w1` key shapes, which these do not match.

When you meet one: report it as blocked, name the strategy, and stop. Do not attempt a hand remap.

## Tool (FSDP only)

`skyrl-train/scripts/consolidate_sharded_checkpoint.py` — reassembles the per-rank shards into
full tensors and writes an HF safetensors directory. CPU only, no process group, no GPUs.

**Always `--inspect` first.** It reports world size, parameter count, dtypes and every distinct
shard placement, and writes nothing. Reassembly depends on the mesh the run used, so read the
report before converting.

```bash
cd skyrl-train
uv run python scripts/consolidate_sharded_checkpoint.py --src <ckpt>/global_step_N/policy --inspect
```

Then convert:

```bash
uv run python scripts/consolidate_sharded_checkpoint.py \
  --src <ckpt>/global_step_N/policy \
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
