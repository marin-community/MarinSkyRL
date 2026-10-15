# DPO: Direct Preference Optimization

Direct Preference Optimization (arXiv:2305.18290) trains on dataset-supplied chosen/rejected
completion pairs instead of generated rollouts. MarinSkyRL runs DPO through the unified
objective stack: `+algorithm_recipe=dpo` selects the loss, and
`environment.env_class=preference_pair` selects the static pair runner, which tokenizes the
dataset's completions and emits adjacent chosen (+1) / rejected (-1) rows per prompt. No
inference engines are constructed, so the full GPU budget serves the policy and the frozen
reference model.

## Data

A preference parquet with a chat `prompt` plus `chosen` and `rejected` text (or assistant
message lists):

```
uv run examples/dpo/ultrafeedback_dataset.py --output_dir "$HOME/data/ultrafeedback"
```

For the DPO paper's IMDb sentiment experiment (GPT-2 + siebert sentiment classifier), see
`examples/dpo/imdb_sentiment_dataset.py`.

## Training

```
bash examples/dpo/run_dpo.sh                       # single-GPU Qwen2.5-0.5B smoke
NUM_GPUS=8 MODEL=alignment-handbook/zephyr-7b-sft-full BETA=0.01 LR=5e-7 bash examples/dpo/run_dpo.sh
```

The Zephyr-7B-β recipe is β=0.01, lr 5e-7 cosine, warmup ratio 0.1, global batch 128, 1 epoch,
max length 1024/512 (alignment-handbook `recipes/zephyr-7b-beta/dpo/config_full.yaml`).

## Reading the metrics

- `policy/dpo/loss` — the true DPO loss, `-log sigmoid(beta * (chosen - rejected) log-ratios)`.
- `policy/dpo/accuracy` — fraction of pairs where the policy's β-scaled margin favors chosen.
- `policy/dpo/chosen_reward`, `policy/dpo/rejected_reward` — β·(log π − log π_ref) sums.
  Standard DPO typically *decreases* chosen rewards while widening margins (likelihood
  displacement); a rising chosen log-prob usually means the pair is wired backwards.
- `policy/policy_loss` is the per-token surrogate row the optimizer sees (a GSPO-style
  linearization with the exact DPO gradient), not the literal loss; read `policy/dpo/loss`
  for the true value.

## Constraints

`validate_dpo` enforces the pairing invariants: `n_samples_per_prompt=2`, even
`micro_train_batch_size_per_gpu`, pair-aligned data-parallel shards, no sample packing,
sequence/context parallelism of 1, no trajectory selector, a frozen reference (no
reference-update callbacks), and `trainer.placement.colocate_all=false`.
