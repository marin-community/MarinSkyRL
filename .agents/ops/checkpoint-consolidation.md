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

### megatron runs: use the export job, not an offline converter

`MegatronStrategy.save_hf_model` converts through `bridge.save_hf_weights` (mbridge), which needs
the Megatron runtime and a live process group — it cannot run on a laptop. A megatron checkpoint
also needs **two** remaps, not one: Megatron-native layer-stacked keys to HF
(`decoder.layers.self_attention.linear_qkv.weight` is `(48, 5120, 2048)`), and the grouped expert
tensors (`decoder.layers.mlp.experts.experts.linear_fc1.weight`, `(48, 128, 1536, 2048)`) to
per-expert HF layout. `--defuse-moe` handles only the second, and only for `.moe.experts.` /
`.experts.w1` key shapes, which these do not match.

**Use `cloud/iris/export_hf_checkpoint.py`.** It does not reimplement the conversion. It re-runs
the trainer's own export as a short iris job: resume the chosen step with `max_steps` set to that
same step, which takes `_handle_resume_at_max_steps`, fires `on_train_end`, and runs
`save_models` → `save_hf_model` → `bridge.save_hf_weights` before exiting 0. No training step runs.

```bash
python -m cloud.iris.export_hf_checkpoint \
  --ckpt_path s3://.../<job>/checkpoints --step 75 \
  --rl_config cloud/iris/configs/<the config the run trained with>.yaml \
  --model_path <base model, as at training time> \
  --num-nodes 4 --gpus-per-node 8      # MUST match the training geometry
```

The geometry must match: the checkpoint is sharded to the parallel layout the run used, and
`bridge.save_hf_weights` gathers across that same mesh. `--dry-run` prints the launch command
without submitting. The export lands in `export_path` on durable storage; publishing to the Hub
stays a separate, owner-authorized step.

Do not attempt a hand remap of Megatron-native tensors.

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


## Publishing a checkpoint by hand

The canonical route, in order. Steps 1-2 are the only supported way to turn a training checkpoint
into weights; there is no offline converter for megatron.

**1. Export the chosen step.** Not the newest — the one trailing-EMA selection picked.

```bash
python -m cloud.iris.export_hf_checkpoint \
  --ckpt_path s3://marin-us-east-02a/iris/<job>/checkpoints --step <N> \
  --rl_config cloud/iris/configs/<the config the run trained with>.yaml \
  --model_path <base model, as at training time> \
  --num-nodes 4 --gpus-per-node 8
```

The geometry MUST match training. `--dry-run` prints the launch command without submitting. The
export lands at `s3://marin-us-east-02a/iris/<job>/exports/global_step_<N>/policy/`.

**2. Confirm it exists before promising anything.** List that prefix and check for
`model-*.safetensors` plus `config.json`. An export job that exits 0 having written nothing is the
failure mode this whole path exists to fix.

**3. Stage flat.** Hub model files sit at the ROOT of the uploaded directory, never under
`policy/`. Copy the export's contents to the root of a clean staging dir, then add the resolved
launch config and `training_logs/` beside them.

**4. Scan for secrets.** The capability JWT above is present in run artifacts and no scanner binary
is installed on the launch host — match a JWT-inside-URL shape explicitly.

**5. Upload.**

```bash
hf upload <namespace>/<repo> <staging_dir> --repo-type model
```

`hf upload`, never `hf upload-large-folder` (a deprecated stub that deadlocks on Hub LFS 429s).
Never `huggingface_hub.upload_folder()` without `delete_patterns=[]`. `--private` takes no value.
Wrap a long upload in `tmux`, not `nohup`.

**Namespace: `laion`** is the default for model repos in this campaign. Note the asymmetry —
`trace_upload.repo_org` is `DCAgent` for trace datasets, so the two do not match and neither is
wrong. Repo naming: `<run-slug>-step<N>-<size>`, with the size taken from the checkpoint's own
tensor shapes rather than from the job or base-model name.

**6. Verify the published repo lists both the weights and `training_logs/`** before deleting any
staging directory or reclaiming any checkpoint.
