# Open-MOPD patched-verl fidelity control

## Overview

This launcher runs the authors' patched `verl` implementation at Open-MOPD commit
`4809a96cf85a869106ff0ff3f37d0a51e12010ae`. It is a control for comparing a future native MarinSkyRL
implementation against the released code. It does not establish native MarinSkyRL fidelity.

The checked-in config pins the MixSFT student, three domain teachers, `rl_prompt_mix` dataset revision, and dataset
SHA-256. The Iris task records the resolved training command, `pip freeze`, `nvidia-smi -q`, source status, and the
complete config in `control-manifest.json`. It uploads that manifest before training, periodically copies the output
tree to durable storage, and performs a final copy when the training process exits.

Future launches stage the pinned 30-row AIME 2024 evaluation parquet separately
from the training mix. The authors' trainer samples one AIME response at
temperature 0.6 after each two-step checkpoint (after step 1 for the one-step
gate). Console metrics and `validation/<step>.jsonl` are copied with the output
tree. This inline check uses the training engine's 16,384-token response cap
and its rule-based math scorer. It is an early quality signal, not the released
64-sample, 31,000-token AIME evaluation; continue to run the separate formal
checkpoint suite for that comparison. The setting applies when launching from
this configuration; it does not retroactively change existing jobs.

## Dry run

The task image must contain Python 3.12, PyTorch 2.8.0, vLLM 0.11.0, Ray 2.55.1, Transformers 4.57.6,
protobuf 3.20.3, flash-attn 2.8.1, flashinfer-python 0.3.1, `huggingface_hub`, `fsspec`, the output URI's fsspec
backend, Git, and CUDA. The task checks the versions before downloading model weights. Use an image digest, not a
mutable tag. The Open-MOPD release does not provide that digest, so the image is an explicit launch input.

```bash
uv run --frozen python -m cloud.iris.open_mopd_fidelity \
  --cluster-config "$IRIS_CLUSTER_CONFIG" \
  --output-uri "s3://<regional-bucket>/experiments/open-mopd/control/one-step" \
  --task-image "<registry>/open-mopd@sha256:<digest>"
```

The command prints structured JSON and the exact `iris job run` command. It does not submit by default. Review the
source commit, artifact revisions, A100×8 request, output prefix, image digest, and known deviations in the printed
plan.

Submission is an explicit second action:

```bash
uv run --frozen python -m cloud.iris.open_mopd_fidelity \
  --cluster-config "$IRIS_CLUSTER_CONFIG" \
  --output-uri "s3://<regional-bucket>/experiments/open-mopd/control/one-step" \
  --task-image "<registry>/open-mopd@sha256:<digest>" \
  --submit --allow-known-deviations
```

Submit only from a clean, committed checkout under the standard Iris launch procedure. Iris always bundles the
workspace, so the task module and pinned config reach the task even with `--no-sync`. That flag disables Iris setup
scripts, preventing the default `uv sync` from replacing the custom image's pinned PyTorch and vLLM environment. The
recorded launcher commit identifies the reviewed workspace bytes.

Use a new output prefix for every run. The launcher disables Iris task retries so a failed optimizer process cannot
silently restart against a partially written local checkpoint.

## Gates

`--gate one_step` is the default. It retains the published batch size of 1,024 and validates one complete
rollout/teacher/update/checkpoint cycle.

`--gate paper_checkpoint` runs 200 steps. The released final model card identifies its checkpoint as step 200, so
this is the first result-bearing comparison.

`--gate paper_schedule` runs the 600 total steps specified in paper Table 8. Run it only after the one-step and
200-step controls have durable manifests and checkpoints.

## Pinned training settings

The resolved authors' command uses one A100×8 node, batch size 1,024, minibatch size 256, one PPO epoch, learning rate `1.5e-6`,
PPO clipping `0.2/0.28`, no KL penalty, one response per prompt, temperature 1, student top-k 16, and nucleus
truncation `p=0.99`. The domain sampler uses the published 2:2:1 math/code/IF ratio. The loss targets equal
one-third domain shares and uses anchored forward gap-following with alpha 1. The 1,024/256 minibatch split and one
PPO epoch produce four optimizer updates per rollout batch. Reward refresh remains enabled and begins after the first
minibatch update. The multi-teacher prompt cap is 2,048 tokens for every domain. The 1,024-token math prompt cap in
Table 8 belongs to the separate single-domain RouteOPD control.

## Known fidelity constraints

- The released patched `verl` tree exposes one global response limit. New launches patch synchronous rollout to cap
  each IF request at 2,048 tokens while retaining 16,384 for math/code; the 16,384-token padded batch shape is
  unchanged. The source patch is recorded in `control-manifest.json`. Earlier reference jobs launched before this
  change used 16,384 for every domain and remain distinct controls; they are not paper-exact on IF response length.
- The release includes partially pinned installation scripts, not a complete lockfile or digest-addressed runtime
  image. A reviewed image digest is required before submission. Preserve its Dockerfile or build record alongside
  the experiment.
- The released final model card says step 200, while Table 8 says 600 total steps. Both gates are exposed and must
  remain separate in result reporting.
- The task downloads about 28 GB of model weights and 936 MB of training data from Hugging Face. Mirror these
  inputs into the selected Iris region before a production run if direct egress is not approved. The current task
  stages directly from the pinned Hugging Face revisions.
- Iris may not have an A100×8 scale group in the selected cluster. Substituting H100×8 changes the hardware control
  and must be recorded as a deviation. Pass `--gpu-slice H100x8` for that schedulable path; the dry-run plan records the
  override before submission.

The acceptance record should include the printed plan, task-image digest, Iris job ID, `control-manifest.json`,
checkpoint inventory, terminal logs, and the exact evaluation protocol from the released `eval.sh`.
