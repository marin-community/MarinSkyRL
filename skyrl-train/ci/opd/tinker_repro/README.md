# Tinker rank-128 reasoning reproduction

This directory contains Tinker-hosted and native MarinSkyRL harnesses for the
public Tinker reasoning result. The hosted path runs OpenThoughts3 SFT, DeepMath
OPD, and AIME 2024 sampling through Tinker. The native path trains and evaluates
the same pinned Qwen3.5 model family on Iris GPUs.

## Training stages

The training launcher exposes closed stages so a smoke-test override cannot be
mistaken for the published configuration.

| Stage | Work | Purpose |
| --- | ---: | --- |
| `sft_plumbing` | 1 batch, 128-row shuffle buffer | Check credentials, data conversion, training, checkpointing, and artifact writes cheaply. Not fidelity-equivalent. |
| `sft_fidelity_step` | 1 batch, 384,000-row shuffle buffer | Exercise the published SFT data path and its large streaming-shuffle buffer. |
| `sft_full` | 3,000 batches | Published rank-128 SFT configuration. Requires an explicit $10,000 maximum-cost acknowledgement. |
| `opd_plumbing` | 1 prompt, 1 rollout, 256 generated tokens | Check the student/teacher path cheaply. Not fidelity-equivalent. |
| `opd_fidelity_step` | 512 prompts, 4 rollouts each, at most 16,384 generated tokens | Exercise one exact-shape published OPD step. Requires an explicit $150 planning acknowledgement. |
| `opd_full` | 200 exact-shape steps | Published rank-128 OPD configuration. Requires an explicit $30,000 maximum-cost acknowledgement. |

The acknowledgement is an authorization record, not a server-enforced Tinker
spending limit. Tinker billing data can lag by hours. Review billing from each
smaller gate before approving the next stage; never submit the full SFT or OPD
stage merely because the planning amount is affordable. Prompt processing,
checkpoint storage, and other charges can make the final bill differ.

Every command defaults to a dry run. For example:

```bash
RUN_ID='tinker-repro-20260914'
OUTPUT_URI="s3://marin-us-east-02a/iris/cw-rno2a/experiments/tinker-opd-repro/${RUN_ID}/sft"

uv run --frozen python skyrl-train/ci/opd/tinker_repro/submit_training_iris.py \
  --stage sft_plumbing \
  --run-id "$RUN_ID" \
  --output-uri "$OUTPUT_URI" \
  --secrets-env "$HOME/Documents/secrets.env"
```

After reviewing and preserving the dry-run JSON, add `--submit`. A full SFT
submission additionally requires `--acknowledge-cost-usd 10000`; an OPD
fidelity step requires `--acknowledge-cost-usd 150`; and a full OPD submission
requires `--acknowledge-cost-usd 30000` and the reproduced SFT state checkpoint:

```bash
uv run --frozen python skyrl-train/ci/opd/tinker_repro/submit_training_iris.py \
  --stage opd_full \
  --run-id "$RUN_ID" \
  --output-uri "s3://marin-us-east-02a/iris/cw-rno2a/experiments/tinker-opd-repro/${RUN_ID}/opd" \
  --sft-checkpoint 'tinker://REPRODUCED_RUN/weights/final' \
  --acknowledge-cost-usd 30000
```

This still prints a dry run. Adding `--submit` is the distinct action that
creates the Iris job.

The submitter loads `--secrets-env` and reads `TINKER_API_KEY` and
`WANDB_API_KEY` only after `--submit`; `HF_TOKEN` is optional. The path defaults
to `$OT_AGENT_SECRETS_ENV` when set, otherwise to `~/Documents/secrets.env` when
that file exists. The parser accepts `KEY=VALUE` and `export KEY=VALUE` lines
without executing the file. Secret values are placed in Iris `EnvironmentSpec`,
not the process arguments. Jobs run directly on `cw-rno2a`, request no accelerator,
are non-preemptible, and have zero automatic retries. Training and AIME 2024
evaluation use interactive priority. SFT requests 4 CPU cores,
32 GB memory, and 50 GB disk because its 384,000-row streaming shuffle buffer
has not yet been measured on Iris; OPD requests 2 CPU cores, 8 GB memory, and
20 GB disk.

The worker refuses a reused S3 prefix, writes a typed redacted manifest before
starting the recipe, and mirrors the local log directory every minute and on
exit. It requires the final `checkpoints.jsonl` record to contain both a Tinker
training-state URI and sampler URI at the expected step. Cookbook periodic
checkpoints expire after seven days, while final checkpoints have no TTL and
continue to incur storage charges until an authorized cleanup changes them.

The worker runtime is isolated from the MarinSkyRL root environment. The PEP
723 lock beside `run_training.py` pins the reviewed Cookbook commit and its
complete transitive environment. Regenerate it only when intentionally changing
the reproduction runtime:

```bash
uv lock --script skyrl-train/ci/opd/tinker_repro/run_training.py
```

The public recipes do not accept Hugging Face dataset revisions. A local adapter
binds every OpenThoughts3 and DeepMath `load_dataset` call to the reviewed
revision recorded in the plan. The worker also verifies that revision is still
the repository head before making a Tinker request. The revisions used for the
originally published runs were not disclosed, so these pins reproduce the data
reviewed for this harness rather than claiming an unknown historical snapshot.

The pinned Cookbook's OPD CLI parses the configured `temperature` but does not
forward it to the training config. The same local adapter forwards that parsed
value explicitly instead of relying on the underlying config default; the
current reproduction plan configures 1.0. Treat a change to either pin or
temperature as a fidelity review, not an automatic dependency update.

## Native MarinSkyRL OPD

`native_opd.py` runs the same published Qwen3.5 student, teacher, LoRA shape,
and reverse-KL objective on eight local GPUs. Its exact vLLM compatibility
backport edits the installed Python source, so Iris jobs must use a task-private
uv cache and copy-mode installation:

The rollout engine reserves 90% of each assigned GPU for weights and KV cache
and admits at most 512 concurrent sequences. The bound reduces repeated
preemption of 16,384-token responses while preserving the published batch and
sampling contract. Teacher prompt-logprob scoring retains a separate 70% GPU
memory reservation and token budget so its float32 log-softmax has transient
memory headroom.

```bash
uv run iris --cluster cw-rno2a job run \
  --enable-extra-resources --gpu H100x8 --no-sync \
  -- env UV_CACHE_DIR=/tmp/tinker-native-uv-cache UV_LINK_MODE=copy \
  uv run --frozen --extra fsdp --extra vllm python \
  skyrl-train/ci/opd/tinker_repro/native_opd.py \
  --stage plumbing --adapter-uri "$ADAPTER_URI" --output-uri "$OUTPUT_URI"
```

The runner refuses a symlinked vLLM source tree rather than modifying Iris's
shared uv cache. `--no-sync` is required because Iris's managed setup currently
hardcodes symlink mode before applying job environment overrides. Use a unique
output URI for every attempt.

`native_aime24.py` evaluates an SFT or OPD LoRA adapter with MarinSkyRL's AIME
environment. It uses the same pinned 30-problem dataset, system prompt,
temperature 1.0, top-p 1.0, disabled top-k, one sample per problem, and 64,000
generated-token limit as the Tinker evaluator. The manifest reports accuracy in
addition to MarinSkyRL's centered `+1/-1` reward mean. A full run fails if any
response reaches the generation limit.

Pass the exact `lora_adapter` checkpoint prefix, not the parent checkpoint or
experiment prefix:

```bash
uv run iris --cluster cw-rno2a job run \
  --enable-extra-resources --gpu H100x8 --no-sync \
  -- env UV_CACHE_DIR=/tmp/tinker-native-aime-uv-cache UV_LINK_MODE=copy \
  uv run --frozen --extra fsdp --extra vllm python \
  skyrl-train/ci/opd/tinker_repro/native_aime24.py \
  --stage smoke --adapter-uri "$ADAPTER_URI" --output-uri "$OUTPUT_URI"
```

Use a new output URI and change `--stage smoke` to `--stage full` after the
single-problem smoke run completes. Evaluation-only LoRA runs reject remote
engines, non-vLLM backends, and missing local adapter directories. These checks
prevent `main_generate` from silently evaluating the base model.

## AIME 2024 evaluation

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

## Evaluation requirements

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
  --max-examples 1 \
  --secrets-env "$HOME/Documents/secrets.env"
```

The submitter prints a secret-free JSON plan and does not submit by default. Review it, then repeat the command with
`--submit`. It loads the same `--secrets-env` default described above and reads `TINKER_API_KEY` only after the
dry-run gate.

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
  skyrl-train/tests/cpu/ci/test_tinker_opd_training.py \
  cloud/iris/tests/test_tinker_opd_aime24_submission.py \
  cloud/iris/tests/test_tinker_opd_training_submission.py
```

References:

- [Tinker distillation recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)
- [Tinker sampling defaults](https://github.com/thinking-machines-lab/tinker/blob/main/src/tinker/types/_pydantic_types/sampling_params.py)
- [AIME 2024 dataset](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
