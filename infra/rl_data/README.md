# RLVR dataset preparation

`python -m infra.rl_data` turns a pinned RLVR source into a local SkyRL train/validation
artifact. Run `python -m infra.rl_data --help` for the current source list.

Run it from the repository root in the `skyrl-train` environment so the selected tokenizer,
`datasets`, and `skyrl-gym` verifier contracts are available:

```bash
uv run --project skyrl-train python -m infra.rl_data \
  --source rlvr_math --revision <source-commit> \
  --validation-source math500 --validation-revision <validation-commit> \
  --tokenizer <training-tokenizer> --max-prompt-tokens 4096 \
  --output-dir /shared/rl-data/rlvr-math
```

The command writes `train.parquet`, `validation.parquet`, and `provenance.json` by staging
the complete directory next to `--output-dir` and renaming it only after both parquet writes
finish. It will not overwrite an existing artifact.

`provenance.json` records source revisions, source and verifier identifiers, prompt-token
statistics, raw/unique/emitted counts, dedup and subsampling limits, and verifier-preflight
coverage. AIME sources use two-sided verifier checks. IFEval validates the constraint schema;
the source does not carry canonical satisfying responses for arbitrary constraints. Code sources
normalize their source-specific tests into the LiveCodeBench runtime schema. They deliberately do
not execute downloaded reference solutions during preparation because the verifier is not a security sandbox.

The DAPO adapter streams by default and stops after 20,000 unique prompts unless
`--unique-cap` overrides it. DAPO's default minimum of 1,000 unique prompts catches cleanup
regressions that collapse the dataset to a shared instruction prefix. MATH-500 is rejected as
a training source unless `--allow-train-on-test` is explicit.
