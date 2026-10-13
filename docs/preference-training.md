# Preference training and task validation

Select `algorithm_recipe=dpo` and set `data.preference_pair_format=text`.
Each training row is one chosen/rejected pair. Set
`generator.n_samples_per_prompt=2`; the runner emits the chosen completion
first and the rejected completion second.

## Data formats

`data.preference_pair_format=text` loads a `prompt` column containing chat
messages with `role` and `content`, plus `chosen`/`rejected` text completions.
A message-list completion may contain one assistant message.
Multiple assistant turns require the tokenized format.

`data.preference_pair_format=tokenized` requires four list columns:

- `chosen_input_ids` and `rejected_input_ids`: complete conversation token IDs.
- `chosen_assistant_masks` and `rejected_assistant_masks`: aligned binary loss
  masks, with 1 for supervised tokens and 0 for context or tool observations.

Both branches must share a nonempty, unmasked initial token prefix. Each branch
must contain at least one supervised token and fit within
`trainer.max_prompt_length + generator.sampling_params.max_generate_length`.
The loader preserves the IDs and masks. Invalid tokenized pairs raise an error;
they are not truncated or silently removed. Tokenize with the training model's
pinned tokenizer and chat template before loading these rows.

## Generated validation

`trainer.evaluation_runner=training` reuses the training runner. Offline DPO
needs a separate generation runner to measure task performance:

- `task` loads prepared TaskSpec JSON from the `task_spec` column in `data.val_data`.
  See [prepared task loading](../skyrl-train/skyrl_train/dataset/prepared_tasks.py).
  Prepare Harbor directories or packed archives before the training command.

Task validation uses the shared rollout engine. Harbor tasks use the Shellbox session.
Selecting task validation makes the role plan include the inference role.
Configure its engine count and parallelism through the generator fields. DPO
synchronizes updated policy weights before evaluation, without synchronizing
after every training update. Validation has a separate `validation/` trace
retention path when it uses a separate runner.

Configure evaluation timing through the evaluation callback or the standard
`trainer.eval_interval` and `trainer.eval_before_train` settings.

The `best_checkpoint` callback accepts `initial_model_uri`, `expected_tasks`,
and `minimum_scored_tasks`. The initial URI identifies the untrained checkpoint
when evaluation runs at step zero. Selection requires exactly
`expected_tasks` returned rows and at least `minimum_scored_tasks` verified
rows; a failed coverage check leaves the previous selection unchanged.
It selects the highest mean [normalized verifier score](reward-metrics.md).
Verified zero scores enter that mean; unscored rows do not.
Ties retain the earlier checkpoint.
It writes `best_checkpoint.json` under `trainer.ckpt_path`, preserves the
selected native checkpoint during cleanup, and reads the previous selection
when training resumes. Use one sample per validation task when these counts
represent task coverage.

## Model loading and trace replay

`generator.model_loading=stage_local` stages the immutable policy artifact's
weights on each Iris node before Ray startup. The default `stream` mode stages
metadata and leaves weight shards remote.

`context_budget.serving_window_tokens` can exceed
`context_budget.request_window_tokens`. This increases server capacity without
increasing client or training sequence limits. If `serving_window_tokens` is
omitted, it takes the value of `request_window_tokens`.

Retained trajectory archives include `prompt_token_ids` in each
response step boundary. These IDs preserve the exact context for each step,
including steps whose conversation context changed. This is retained evidence
for replay consumers. See [trajectory retention](trajectory-retention-design.md)
for retention configuration and archive contents.
