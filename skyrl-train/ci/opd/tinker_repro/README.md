# Tinker OPD AIME 2024 evaluation

This evaluator measures a Tinker sampler checkpoint on the 30-problem
`HuggingFaceH4/aime_2024` dataset. It pins the dataset revision and uses the
Tinker Cookbook AIME prompt, answer extraction, renderer, trajectory storage,
and aggregation.

The published Qwen3.5 OPD recipe reports approximately 76.7% on AIME 2024 with
temperature 1.0, top-p 1.0, top-k disabled, and at most 64,000 generated tokens.
It does not publish the evaluation sample count, seed, dataset revision, or
complete evaluator. Since 76.7% is 23 correct answers out of 30, this harness
uses one sample per problem by default. This is the closest documented
calibration, not a claim that every stochastic rerun will score exactly 76.7%.

## Requirements

Create a Tinker API key and export it as `TINKER_API_KEY`. The released sampler
checkpoint must be visible to that Tinker account. `HF_TOKEN` is optional for
this public dataset and tokenizer but is useful when Hugging Face applies rate
limits.

The Tinker Cookbook dependency is deliberately absent from the MarinSkyRL root
environment. Its Transformers constraint conflicts with the root training
environment, so the evaluator carries PEP 723 script metadata and a dedicated
uv lockfile pinned to the reviewed cookbook revision. The evaluator records the resolved versions of
Tinker, Transformers, Datasets, and Tinker Cookbook in its JSON summary, and
fails before sampling if Tinker's top-p or top-k defaults differ from the
published evaluation settings.

```bash
CHECKPOINT='tinker://de58946a-6bfd-5ab2-821f-03b61d237b5b:train:0/sampler_weights/final'
OUTPUT_DIR="$PWD/artifacts/tinker-opd-aime24"

uv run --locked --script skyrl-train/ci/opd/tinker_repro/evaluate_aime24.py \
  --checkpoint "$CHECKPOINT" \
  --save-dir "$OUTPUT_DIR"
```

Start with `--max-examples 1` before paying for the complete evaluation. A full
comparable run must contain all 30 examples with zero API errors and zero
truncations. Under those conditions, `score` and `score_completed` are equal.
When `--num-samples` is greater than one, `score` is mean sample accuracy and
the result also contains pass@k estimates; pass@k is not comparable to the
published 76.7% figure.

The output directory contains `aime_2024/trajectories.jsonl` and
`aime_2024/result.json`. The command also prints a JSON summary containing the
checkpoint, immutable dataset identity, sampling parameters, raw score, and
completed-only score.

## Iris

Iris only orchestrates this job. Tinker hosts model sampling, so the task needs
CPU and network access but no accelerator. Use the SDK submitter instead of
`iris job run -e`: placing the key in a CLI argument exposes it through the
submitter's process arguments. The helper reads `TINKER_API_KEY` from its own
environment and places it directly in the Iris `EnvironmentSpec`:

```bash
CHECKPOINT='tinker://de58946a-6bfd-5ab2-821f-03b61d237b5b:train:0/sampler_weights/final'
OUTPUT_DIR='s3://marin-us-east-02a/iris/cw-rno2a/experiments/tinker-opd-aime24/released-rank128'

uv run --frozen python skyrl-train/ci/opd/tinker_repro/submit_iris.py \
  --checkpoint "$CHECKPOINT" \
  --save-dir "$OUTPUT_DIR" \
  --max-examples 1
```

The submitter prints a secret-free JSON plan and does not submit by default. Review it, then repeat the command with
`--submit`. The submitter reads `TINKER_API_KEY` only after the dry-run gate.

The submission goes directly to `cw-rno2a` at interactive priority. It requests
2 CPU cores, 8 GB of memory, and 20 GB of disk on a non-preemptible worker, with
zero task retries. The API key is never included in the evaluator command or job
entrypoint. Do not print the `EnvironmentSpec`, which necessarily contains the
key sent to the worker.

Every run requires a new, empty `--save-dir`. The evaluator claims it with `reproduction-manifest.json` before the
first sampling request and replaces that file with the complete result after validation. This prevents the cookbook's
resume behavior from mixing samples from different reproduction attempts.

Remove `--max-examples 1` only after the smoke test confirms checkpoint access,
object-store writes, renderer behavior, and expected billing.

## Tests

The focused CPU tests cover the pinned external data contract without installing
Tinker Cookbook into the root environment:

```bash
uv run --frozen --extra cpu --group dev --group harbor-test pytest \
  skyrl-train/tests/cpu/ci/test_tinker_opd_aime24_evaluator.py \
  cloud/iris/tests/test_tinker_opd_aime24_submission.py
```

References:

- [Tinker distillation recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)
- [Tinker sampling defaults](https://github.com/thinking-machines-lab/tinker/blob/main/src/tinker/types/_pydantic_types/sampling_params.py)
- [AIME 2024 dataset](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
