# Snowball and Grug PivotRL comparison

## Method and preregistration

Run SWE and Terminal separately for each student in
[the artifact registry](../cloud/iris/configs/snowball_pivotrl_artifacts.json).
Use the pinned initial student as the frozen profiling policy and KL reference.
Never select examples using NVIDIA's profiling columns.

[PivotRL](https://arxiv.org/html/2603.21383v1) Eq. 5 selects prefixes with positive
reward variance and mean success below a difficulty threshold. Algorithm 1 freezes
this selection before optimization. Eq. 7 penalizes KL(policy || initial reference).
The threshold is explicitly defined as `lambda_diff`; a numerical setting has not
been located in Eq. 5, Algorithm 1, or Appendix A.2–A.3. Keep its value explicit
pending verification of the cited setting. Report all three preregistered KL coefficients: 0, 0.001,
and 0.01. Use eight frozen-policy samples per candidate, binary outcomes, no shaping,
and no online replacement. The implemented KL is a conditional token forward-KL
estimator, importance weighted under the behavior policy, with differentiable weights.

`pivot.mode: pivotrl` uses sampled GRPO actions. `pivot.mode: sft` uses standard
next-action cross entropy on the released demonstration, with one target per prefix
and no KL. Both modes consume the **same student-selected train.parquet**. An SFT
run on all candidates is a separate data-selection ablation and must be labeled so.

The released data supports a method comparison, not an exact paper reproduction.
The requested default SWE verifier compares tool names and arguments; the paper's
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
duplicate repetitions, validation contamination, nonbinary outcomes, and verifier
failures. Context-overflow exclusions are recorded separately from incorrect answers.
Outputs include every candidate's student mean/variance in
`profiled_candidates.parquet`, `statistics.jsonl`, the selected `train.parquet`, and a
manifest with selected/rejected/excluded counts and profiling provenance. Filtering
will reduce row counts; it must not duplicate selected rows to match the release.

## Training and evaluation

The controlled pilot uses [the 96-GPU recipe](../cloud/iris/configs/snowball_pivotrl_96gpu.yaml),
with the user-specified six-stage policy, four-stage reference, and CPU optimizer
offload settings. Select it with `pivot_recipe --template`. Both methods receive
`trainer.loss_token_budget: 1000000`, constant learning rate with no warmup, one
update epoch per batch, and identical selected data. The epoch/step ceilings are
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
uv run python -m infra.rl_data.pivot_report "$ONE_CHECKPOINT_HELDOUT_ARCHIVES"
```

The accuracy report bootstraps source trajectories for 95% intervals, preserving
dependence between neighboring prefixes. Its effective independent units are the
19/20 heldout trajectories, not 262/256 independent tasks. Report one checkpoint's
evaluation at a time. Infrastructure/verifier failures must be repaired, not scored
as incorrect predictions.

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
