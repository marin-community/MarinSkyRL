# RLVR dataset preparation

`python -m infra.rl_data` turns a pinned RLVR source into a local SkyRL train/validation
artifact. Run `python -m infra.rl_data --help` for the current source list.

Run it from the repository root so the selected tokenizer, `datasets`, and `skyrl-gym`
verifier contracts are available from the root environment:

```bash
uv run python -m infra.rl_data \
  --source rlvr_math --revision <source-commit> \
  --validation-source math500 --validation-revision <validation-commit> \
  --tokenizer <training-tokenizer> --max-prompt-tokens 4096 \
  --output-dir /shared/rl-data/rlvr-math
```

The command writes `train.parquet`, `validation.parquet`, and `provenance.json` by staging
the complete directory next to `--output-dir` and renaming it only after both parquet writes
finish. It will not overwrite an existing artifact.

Use `--mixture` to compose independently validated sources with different verifier environments:

```yaml
train:
  - source: deepscaler
    revision: <dataset-commit>
    cap: 40000
  - source: eurus2_code
    revision: <dataset-commit>
    cap: 24000
  - source: openscience # MCQ substitute until an SCP-116K judge env exists
    revision: <dataset-commit>
    cap: 25000
  - source: reasoning_gym
    revision: 0.1.25
    parameters:
      tasks: [leg_counting, knights_knaves]
      rows_per_task: 18500
      seed: 42
  - source: nemotron_if
    revision: <dataset-commit>
    cap: 10000
validation:
  - source: aime24
    revision: <dataset-commit>
  - source: math500
    revision: <dataset-commit>
    cap: 100
  - source: reasoning_gym
    revision: 0.1.25
    parameters:
      tasks: [leg_counting, knights_knaves]
      rows_per_task: 100
      start_index: 18500
      seed: 42
  - source: eurus2_code
    revision: <dataset-commit>
    split: validation
    cap: 100
```

```bash
uv run python -m infra.rl_data \
  --mixture mixture.yaml \
  --tokenizer <training-tokenizer> --max-prompt-tokens 8192 \
  --output-dir /shared/rl-data/prorl-mixture
```

Each source is normalized and preflighted against its own verifier before the rows are
deterministically shuffled. Per-source revisions, counts, verifier identifiers, and emitted shares
remain separate in `provenance.json`. Reasoning Gym validation slices use `start_index` to reserve a
deterministic holdout after the training slice. Its revision is the installed `reasoning-gym` package
version rather than a Hugging Face commit.

The math source registry also accepts `hendrycks_math`, `aime_1983_2024`, `asdiv`, `svamp`,
`numina_math`, and `hardmath`. They use the two-sided `aime` verifier contract. Hendrycks MATH accepts
an optional `parameters.subjects` list; source-native MATH levels, ASDiv grades, and NuminaMath source
tags are retained in `extra_info` for downstream sampling.

`provenance.json` records source revisions, source and verifier identifiers, prompt-token
statistics, raw/unique/emitted counts, dedup and subsampling limits, and verifier-preflight
coverage. AIME sources use two-sided verifier checks. IFEval validates the constraint schema;
the source does not carry canonical satisfying responses for arbitrary constraints. Code sources
normalize their source-specific tests into the LiveCodeBench runtime schema. They deliberately do
not execute downloaded reference solutions during preparation because the verifier is not a security sandbox.

Set `environment.skyrl_gym.lcb.reward_mode=fractional` to reward Eurus code trajectories by the
fraction of unit tests passed. The default `binary` mode remains all-or-nothing. Nemotron IF rows
retain all constraints in one example and receive the fraction satisfied. A free-form SCP-116K
judge environment is not yet available; use the existing `openscience` MCQ source for the STEM
slice when an explicit task substitution is acceptable.

The DAPO adapter streams by default and stops after 20,000 unique prompts unless
`--unique-cap` overrides it. DAPO's default minimum of 1,000 unique prompts catches cleanup
regressions that collapse the dataset to a shared instruction prefix. MATH-500 is rejected as
a training source unless `--allow-train-on-test` is explicit.

`gretel_text_to_sql` (`gretelai/synthetic_text_to_sql`, Apache-2.0) is a single-turn text-to-SQL
source on the `text_to_sql` verifier. The prompt carries the question and the `CREATE TABLE`
schema only; the seed `INSERT`s and the reference query travel in `reward_model.ground_truth` as
a JSON object. The verifier rebuilds an in-memory SQLite database from that DDL and scores
result-set equivalence — column count and column order compared, column aliases ignored, row
order compared only when the reference has a top-level `ORDER BY` — on the seeded database and on
a copy with every third row deleted, so a query returning a hard-coded literal fails the second
grade. The candidate query runs read-only under a single-statement check, a keyword blocklist,
and a SQLite authorizer. The source transform is static: rows with a non-`SELECT`, multi-statement,
clock/RNG-dependent, or schema-qualified-DDL reference, or with no `INSERT`s, are dropped during
conversion. Whether the reference query actually executes is checked once per row by the
contract's two-sided `validate_example` preflight against the reference itself and `SELECT 1` —
unlike the code sources, this one can run that preflight because the "program" is a `SELECT`
over an ephemeral in-memory database with no filesystem or network.
