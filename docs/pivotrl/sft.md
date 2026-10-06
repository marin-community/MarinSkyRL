# Supervise released next actions

`PivotSFTRunner` renders the released next action with the base model's chat template,
including its tool schema and end marker. Loss covers the action suffix only. Prompts
and targets are preserved; a target that does not extend the exact generation prefix
or exceeds the context window is rejected. Evaluation uses the sampled runner.

Use the standard training entrypoint with these Hydra overrides on a prepared
Nemotron Ultra parquet dataset:

```text
generator.reference_actions=true
data.prompt_length_policy=keep
trainer.algorithm.policy_loss_type=sft
trainer.algorithm.advantage_estimator=uniform
trainer.algorithm.use_kl_loss=false
trainer.algorithm.use_kl_in_reward=false
trainer.algorithm.use_entropy_loss=false
trainer.algorithm.off_policy_correction=none
generator.n_samples_per_prompt=1
trainer.update_epochs_per_batch=1
trainer.loss_token_budget=1000000
trainer.eval_loss_token_interval=250000
```

The last update masks surplus loss positions to hit the budget exactly. Complete
responses remain available for inspection. Token counts are checkpointed so a resume
continues the same budget. Use a positive evaluation interval to load validation data
and activate the evaluation callback. The experiment configs added with the full
PivotRL workflow specify the model, dataset, resources, and sampling settings.
