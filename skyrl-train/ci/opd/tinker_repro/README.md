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
