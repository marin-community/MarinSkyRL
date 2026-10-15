# Pi shared-engine smoke gates

These are **integration gates**, not model-quality runs. Use the typed Iris launcher
with the source recipes in `cloud/iris/configs/pi_verifier_grpo_smoke.yaml` and
`pi_teacher_trace_smoke.yaml`. Do not submit either recipe as a standalone Iris
job: the launcher supplies the pinned model source, staged local path, placement,
checkpoint paths, and immutable run/attempt identifiers. The recipes compose
through `scenario.preset`; the example recipes request one or two 8×H100
nodes (the second node is the disjoint managed vLLM teacher role). Match the
typed launch's node shape to the target cluster: `cw-us-east-08a` currently
has **4×GB200**, not 8×H100, so set `generator.num_inference_engines: 4`,
`trainer.placement.policy_num_gpus_per_node: 4`, and allocate 4×GB200 per
node there. No GPU job has been launched from these recipes yet.

Before composing a typed launch config, set:

- `PI_SMOKE_TASK_DATA`: one **direct Harbor task source** with at least
  `trainer.train_batch_size` (8) distinct executable tasks and their machines.
  A packed `path`/`task_binary` parquet is extracted to local task directories
  by the Iris launcher; local task directories and other supported Harbor
  sources also work. No Nemotron reference parquet or
  `extra_info.nemotron_ultra.*` routing is needed. The verifier GRPO and
  verified-only SFT presets require functioning task graders; unfiltered SFT
  does not. Never copy a sampled trial to fill a GRPO group. Pi's current
  installed-agent setup downloads apt/nvm/npm dependencies, so the task
  machine must allow sandbox internet access. The smoke recipes select
  Daytona explicitly and require `DAYTONA_API_KEY` **on every rollout worker**.
  The existing Iris launcher resolves the RL-specific Secret Manager key and
  forwards it as `DAYTONA_API_KEY` to task pods. Direct Pi tasks on the standard
  entrypoint were previously missing from that credential-detection path; the
  launch preflight now recognizes `scenario.preset` too. This is a description
  of existing launcher behavior, not authorization to retrieve or use the key.
  CoreWeave's Iris pool does not offer Docker access to these jobs: the Docker
  trial failed before Pi ran with `Unknown Shellbox factory: 'harbor'` when
  no Docker CLI was available to the rollout workers. The preceding Daytona
  trial failed with `DaytonaAuthenticationError`. Do not submit another GPU
  trial until worker-side Daytona authentication is verified.
  For remote packed task parquets,
  put the URI in the typed launch's `inputs.train_data` as well: the training
  driver stages and extracts that input on each node before SkyRL reads it.
  Putting an S3 URI only in the source recipe bypasses staging; PyArrow's
  native S3 client cannot use the worker's fsspec-injected CoreWeave endpoint.
- `PI_SMOKE_OUTPUT`: a durable trajectory URI co-located with the assigned GPU
  zone. Give each run an isolated prefix. Never use a workstation-local path.
- For teacher SFT, `PI_SMOKE_STUDENT_LOCAL_PATH`: the exact
  `inputs.model.local_path` in the typed launch config. The teacher preflight
  rejects a different student path. The ungraded SFT smoke has no student
  evaluation; run a separate student-only evaluation after the training gate.

Pin the policy source and tokenizer in the typed launch's `inputs.model`, stage
these small datasets in the GPU region, and use that region's artifact store.
Avoid any bulk cross-region transfer; get sign-off for transfers over 10 GB.
The Qwen3-0.6B student and pinned Qwen2.5-0.5B independent teacher here are
*integration fixtures*, not a claim that the latter improves the former.
The typed launch's staged model path replaces the policy path in the recipe.
`cloud.iris.launch` also reads/writes its artifact manifest from the launching
machine **even for a PREPARE-only launch**. A headless launcher targeting
CoreWeave must have authorized CoreWeave S3 object-store credentials locally;
Iris IAP credentials alone do not authorize that object store. Do not submit
an unreviewed job or move injected worker secrets to the launcher to bypass
this check.
The verifier's Pi model alias is derived from the served policy, not hardcoded
to a source repository name. Managed teacher weights must resolve at their
pinned revision on the GPU workers; the teacher is not weight-synced from the
student. Confirm the worker image includes the Harbor commit pinned by
`uv.lock` before submission; after the Harbor PR merges, update the dependency
to its maintained upstream revision.

For an offline composition check (without contacting Iris), set the environment
variables above to harmless placeholder paths and run:

```bash
uv run --frozen --extra cpu --group harbor-test pytest -q \
  cloud/iris/tests/test_launch_config.py \
  cloud/iris/tests/test_pi_scenario_composition.py
```

That verifies source-recipe selection, typed-launch staging, policy/teacher
binding, and whole-node placement. It does **not** verify dataset availability,
Iris capacity, vLLM tool-template parity, task execution, or optimizer updates.
Do not treat a successful PREPARE-only launch as a sampled-model smoke.

Run verifier first. Confirm two **independently sampled** Pi trials per prompt
have different trial IDs and valid exact served-token prefixes; record Harbor
verifier outcomes separately from any future LLM judge. Require a group to
survive admission and one finite policy update, then inspect the checkpoint.
For teacher SFT, confirm the teacher produces actions on an independently
allocated engine, that their executed rendering becomes student-tokenizer SFT
labels (no student behavior logprobs on teacher tokens), and that one finite
optimizer step finishes without a verifier result. The config switch is
`scenario.preset: teacher_demo_pi_base` for unfiltered SFT, or
`teacher_demo_pi_verified_base` for grader-gated SFT. The latter accepts only
explicitly passed grades, masks failed trials from training, and fails on
missing or unavailable verdicts. It needs a real verifier-bearing task pack;
neither preset uses the grade as an SFT loss weight. A separate student-only
evaluation can be scheduled with Harbor tasks under the verified preset.
A zero-advantage verifier group is a valid sampled outcome, not evidence of a
learning signal; never duplicate or relabel it to manufacture a nonzero group.

Treat each failure (missing model provenance, tool-rendering prefix mismatch,
unauthenticated external source, worker allocation conflict, absent verifier,
failed task import, or tokenizer mismatch) as a stop-and-diagnose gate. These
short smokes do not establish restart correctness, distributed reliability,
production learning gains, or judge/DAgger/critique readiness. Teacher **soft
token-distribution KL** is a separate probability-scoring objective, not the
teacher-trace SFT smoke.
