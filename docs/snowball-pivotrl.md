# Snowball and Grug PivotRL comparison

## Method and preregistration

Run SWE and Terminal separately for each student in
[the artifact registry](../cloud/iris/configs/snowball_pivotrl_artifacts.json).
Use the pinned initial student as the frozen profiling policy and KL reference.
Never select examples using NVIDIA's profiling columns.

[PivotRL](https://arxiv.org/html/2603.21383v1) Eq. 5 selects prefixes with positive
reward variance and mean success below a difficulty threshold. Algorithm 1 freezes
this selection before optimization. Eq. 7 penalizes KL(policy || initial reference).
The threshold is explicitly defined as `lambda_diff`. Sweep the user-approved values
0.25, 0.5, 0.75, and 1.0, using 0.5 for the initial comparison. These are experiment
settings, not claimed numerical defaults from the paper. With eight profiling samples
and the strict inequality in Eq. 5, they retain 1, 1–3, 1–5, and 1–7 successes,
respectively. Report all three preregistered KL coefficients: 0, 0.001,
and 0.01. Use eight frozen-policy samples per candidate, binary outcomes, no shaping,
and no online replacement. The implemented KL is a conditional token forward-KL
estimator, importance weighted under the behavior policy, with differentiable weights.

The comparison has three arms, each with the same loss-token budget:

| Mode | Training data | Objective |
| --- | --- | --- |
| `sft_random` | `random_train.parquet` | Next-action cross entropy, no KL |
| `sft` | Student-selected `train.parquet` | Next-action cross entropy, no KL |
| `pivotrl` | Identical student-selected `train.parquet` | Sampled GRPO with reference KL |

Random SFT samples uniformly without replacement from the context-eligible training
candidate pool, with seed 42 and the same row count as the selected pivots. It does
not condition on success statistics; overlap with selected pivots is allowed and
reported. The two SFT arms isolate the effect of selection. Selected SFT versus
PivotRL compares objectives on identical data. Heldout trajectories are excluded
before either selection. All arms share initialization, tokenizer, Flash Attention,
context window, optimizer settings, and heldout evaluation. Response lengths, prompt
exposure, and update counts can differ and must be reported.

The released data supports a method comparison, not an exact paper reproduction.
The default SWE verifier compares tool names and arguments; the paper's
appendix describes a tool-name comparator. Terminal uses the released agent's
[Terminus-2 string-only verifier](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/terminus_judge):
JSON schema, completion state, and command-string similarity. Neither verifier
executes the source task or calls a semantic judge.

## Immutable data

The registry contains complete S3 locations, source SHA256 checksums, parquet
checksums, revisions, split seeds, and heldout trajectory IDs. All prepared objects
are under `s3://marin-us-east-02a/marin/users/dml/skyrl/snowball-pivotrl/prepared/`.

| Release | All rows | Training candidates | Heldout rows | Heldout trajectories |
| --- | ---: | ---: | ---: | ---: |
| SWE | 50,661 | 50,399 | 262 | 19 |
| Terminal | 31,111 | 30,855 | 256 | 20 |

The cached raw files matched the pinned Hugging Face LFS checksums. SWE contains
353 more rows than the stated 50,308; all rows, including duplicates, are preserved.
Terminal matches the stated count. `release.parquet` preserves the entire release;
`candidates.parquet` and `validation.parquet` partition it by source trajectory.
The split precedes profiling and is shared across students and methods.

Preparation accepts the existing local cache directly:

```bash
uv run python -m infra.rl_data.pivot prepare --dataset swe \
  --source ~/.cache/marin/datasets/nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1/train.jsonl \
  --output /path/to/new/prepared-swe
uv run python -m infra.rl_data.pivot prepare --dataset terminal \
  --source ~/.cache/marin/datasets/nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1/atcb_terminal_pivot_release_final_v2.jsonl \
  --output /path/to/new/prepared-terminal
```

The adapter retains request options, tools, source identity, expected action/answer,
and original profiling metadata. Only adapted parquet belongs in `PromptDataset`.

## Profiling and selection

Generate each student/domain recipe with its candidate URI and both validation URIs
from the registry:

```bash
uv run python -m infra.rl_data.pivot_recipe --student snowball --mode profile \
  --train-data "$CANDIDATES_URI" \
  --validation-data "$SWE_VALIDATION_URI" "$TERMINAL_VALIDATION_URI" \
  --output snowball-swe-profile.yaml
```

Repeat for `--student grug` and each domain. The profile mode uses generate-only
execution over candidates, streams batches through the standard runner, retains
every outcome, and never updates weights. Keep the pinned model path/revision when
assembling the Iris launch; model overrides must not silently select another policy.
Run a separate bounded candidate slice for the initial profiling memory smoke.

After profiling, stage that run's retention archives locally and select once:

```bash
uv run python -m infra.rl_data.pivot filter --artifacts "$PREPARED_LOCAL" \
  --rollouts "$PROFILE_ARCHIVES_LOCAL" --output "$FILTERED_LOCAL" \
  --profile-policy "$PINNED_MODEL_REPO" --profile-revision "$PINNED_MODEL_SHA" \
  --samples-per-prefix 8 --difficulty-threshold "$PREREGISTERED_THRESHOLD"
uv run python -m infra.rl_data.pivot_publish --artifacts "$FILTERED_LOCAL" \
  --destination s3://marin-us-east-02a/marin/users/dml/skyrl/snowball-pivotrl/
```

Selection rejects wrong policy identities, updated/resumed policies, missing samples,
duplicate repetitions, validation contamination, and nonbinary outcomes. Profiling
continues after retained request/verifier errors. A candidate with any failed attempt
is excluded from both pivot selection and the random control; its partial outcomes
never receive a mean or variance. Every candidate must still have eight retained
attempts, so missing batches cannot silently pass as complete profiling. Error counts
and context-overflow exclusions are recorded separately from incorrect answers.
Outputs include every candidate's student mean/variance in
`profiled_candidates.parquet`, `statistics.jsonl`, the selected `train.parquet`, the
count-matched `random_train.parquet`, and a
manifest with selected/rejected/excluded counts and profiling provenance. Filtering
will reduce row counts; it must not duplicate selected rows to match the release.

## Training and evaluation

The controlled pilot uses [the 96-GPU recipe](../cloud/iris/configs/snowball_pivotrl_96gpu.yaml),
with a six-stage policy, four-stage reference, and CPU optimizer
offload settings. Select it with `pivot_recipe --template`. All three arms receive
`trainer.loss_token_budget: 1000000`, constant learning rate with no warmup, one
update epoch per batch. The selected SFT and PivotRL arms use identical data; random
SFT uses the count-matched control. The epoch/step ceilings are
safety bounds; compare only runs that actually reach the token budget.

The last batch masks surplus loss positions without shortening generated actions.
Token counts are persisted in checkpoints; a budgeted resume without saved counts
fails. Log loss, response, prompt, and input totals; policy/reference logical forward
tokens; rollout generation tokens; optimizer steps; and allocated GPU-hours per
training cycle. Logical forward counts exclude recomputation and padding. Use Iris
allocation timestamps for end-to-end GPU-hours, including startup, profiling, and
evaluation, and report profiling separately for its amortized cost.

Heldout evaluation runs before training, every two updates, after crossing each
250,000-loss-token milestone, and at completion. Milestone evaluations record the
actual consumed count; only the final one-million-token budget is exact. Multiple
milestones crossed by a large batch produce one evaluation at that batch's endpoint.
Keep response lengths and optimizer-update counts visible: equal loss-token budgets
do not equalize these quantities. Use this pilot's throughput to size a longer run.

Generate recipes using the selected immutable train URI. Run `--mode pivotrl` for
each `--kl-coefficient 0`, `0.001`, and `0.01`; run `--mode sft` on that identical URI.
Run `--mode sft_random --train-data <immutable-random_train.parquet-uri>` for the
random control. The filter manifest stores its seed, pool size, count, overlap,
and checksum. Profile the full training candidate splits for each student: 50,399 SWE
rows and 30,855 Terminal rows, after excluding heldout trajectories. Both the pilot
and full training comparisons derive their selected and random-control artifacts
from these complete pools. Small subsets are only for mechanics checks. They must
not define the experimental training pool. Stream profiling in bounded batches;
eight samples per row give up to 403,192 SWE and 246,840 Terminal responses per
student, with context exclusions reported separately. Reuse these frozen profiles
for all four thresholds and all three KL coefficients. Keep both full heldout sets.
Start with KL 0.001; report the preregistered 0 and 0.01 runs as
follow-ups, before making conclusions about KL choice.
Add `--smoke` for one training step before the full arm. The template preserves the
split64 Megatron and vLLM geometry. Batch sizes count prefixes in this trainer:
`train_batch_size: 64`, `policy_mini_batch_size: 64`, and 16 responses give 1,024
trajectories. A mini batch size of 1,024 would incorrectly request 1,024 prefixes.

`max_new_tokens_per_turn: null` is supported only for a single Gym turn. Each request
uses the context window minus its actual tokenized prefix; teacher output caps do
not truncate sampled actions. SFT targets that exceed the window fail explicitly.
The full recipe requests at most 130 steps and one epoch; a small selected dataset
can exhaust that epoch sooner, which must be reported rather than silently resampled.

Check the first PivotRL training step's retention archive before a full run:

```bash
uv run python -m infra.rl_data.pivot_report "$FIRST_TRAIN_STEP_ARCHIVE" --check-geometry
uv run python -m infra.rl_data.pivot_report "$ONE_CHECKPOINT_HELDOUT_ARCHIVES" \
  --heldout-parquet "$SWE_VALIDATION_PARQUET" "$TERMINAL_VALIDATION_PARQUET"
uv run python -m infra.rl_data.pivot_report "$PIVOTRL_HELDOUT_ARCHIVES" --reference "$SFT_HELDOUT_ARCHIVES" \
  --heldout-parquet "$SWE_VALIDATION_PARQUET" "$TERMINAL_VALIDATION_PARQUET"
uv run python -m infra.rl_data.pivot_report "$TRAINING_ARCHIVES" --exposure
```

Accuracy commands require the prepared validation parquet files. Before computing
intervals, the reporter checks every expected source row appears exactly once at
one checkpoint. Missing rows, unexpected rows, repeated predictions, and verifier
infrastructure errors stop reporting. Input-overflow exclusions are counted separately
from verified outcomes; they are not incorrect answers. Paired differences require
identical verified rows in both arms.

Exposure reports distinguish unique source rows, prefix visits, and responses per
row. Their token totals describe retained trajectories before the final loss-budget
mask; the trainer's `consumed/loss_total` is the authoritative loss-token count.
Use W&B team/project `marin-community/pivot-rl`. Heldout aggregate scores appear as `eval/pivot_swe/avg_score` and
`eval/pivot_terminal/avg_score`, mirrored to Iris stdout as `WANDB_MIRROR`.
Per-row predictions and verification results remain in the run's immutable
`attempts/trajectories` archives. Confidence intervals are computed by the report
command above; they are not automatically added to W&B.

The accuracy report bootstraps source trajectories for 95% intervals, preserving
dependence between neighboring prefixes. Its effective independent units are the
19/20 heldout trajectories, not 262/256 independent tasks. Report one checkpoint's
evaluation at a time. Infrastructure/verifier failures must be repaired, not scored
as incorrect predictions.
The paired comparison requires identical source rows and repetition IDs, and
bootstraps whole source trajectories jointly across arms. Its difference interval
therefore preserves shared successes and failures between checkpoints.

After registering each exported checkpoint, run the existing Marin evaluation CLI
from the Marin repository:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model <registered-checkpoint> --evals swebench,tb2 --priority interactive --no-wait
```

Keep the policy's OOD suite untouched until checkpoint and hyperparameters are frozen.
The prepared artifacts and recipes do not establish GPU startup success, first-rollout
geometry, filtered counts, training completion, or evaluation scores; those require
the corresponding runs and their retained evidence.


Profiling retries failed requests up to `generator.pivot_profiling_max_retries` (default 2).
Successful and context-excluded requests are never resampled. Each retained attempt carries
`extra_info.profiling_attempt`, keeping the original source row and repetition ID. Filtering
validates the complete attempt history and uses the terminal verified outcome once, regardless
of archive order. Exhausted failures remain excluded from both training pools and are reported;
they never become zero rewards. Profiling token counts include all returned attempts, and the
summary reports recovered and unresolved samples separately. A fatal runner or retention error
still propagates: retries apply to retained per-request failures, including a batch in which every
request failed, and do not conceal programming errors or loss of the inference engines.


The runtime pins xgrammar 0.2.8, which fixes overridden EOS tokens being allowed
inside unfinished tool JSON ([upstream fix](https://github.com/mlc-ai/xgrammar/pull/905)).
With Grug, generation uses both EOS IDs 128001 and 128009 while the tokenizer declares
only 128001. Earlier xgrammar releases could allow 128009 through the mask and then
reject it in the matcher, producing a server error. The pin preserves strict tool schemas.
