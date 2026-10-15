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
node there. Earlier isolated GPU teacher attempts have not completed an optimizer step.

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
  trial failed with `DaytonaAuthenticationError`. The subsequent authenticated
  Daytona attempt entered the installed-agent run but produced no model-turn
  evidence before its 600-second timeout; the sandbox closed on cancellation,
  so the following log download failed. A shorter live-diagnostic attempt showed
  Pi setup and uploads finishing in four Daytona sandboxes, then each Pi CLI
  call stalling without reaching the model callback. A one-sandbox manual Pi
  trial against a local model stub completed with in-memory capture. Harbor's
  recorder was dropping the path-scoped Iris capability token when forwarding
  `/v1/...` upstream; the pinned Harbor fix preserves that path and closes the
  loopback server if collation fails. A managed GPU retry confirmed the route
  now reaches the model callback, but failed at the next gate: the teacher sent
  one output token (effective `max_completion_tokens: 1`) with no logprobs, so
  the recorder rejected incomplete literal evidence. The cause of the one-token
  budget and the teacher evidence contract need separate diagnosis; no optimizer
  step passed. Do not count the manual stub trial as an optimizer step.

For a targeted follow-up only, add
`terminal_bench.installed_agent.agent.kwargs.verbose_sandbox_diagnostics: true`
to a **fresh disposable launch config**. The default is `false`; do not add the
flag to the committed smoke preset or leave it on after the investigation. Pi
writes `/logs/agent/pi-sandbox-diagnostics.jsonl` inside Daytona with only
fixed-schema budget counts, tokenization status, upstream HTTP status, and
presence/counts of literal evidence. The worker validates the schema before
publishing `engine/pi-sandbox-diagnostics.jsonl` in the trial's output prefix,
even if in-memory collation fails after Pi exits. Neither prompts, outputs,
commands, model IDs, endpoint URLs, credentials, nor raw Pi stdout belong in
this diagnostic file. A killed/closed sandbox may still prevent collection;
the live host-side phase journal remains separate.

One isolated attempt (`attempt-11`) exercised this flag on managed GPUs. Its
sandbox diagnostic confirmed that Pi handed the budget extension a **one-token
configured output cap** *before* the extension ran; the extension's served
`/tokenize` count was 1,484 against a 2,048-token context, leaving 564 actual
output slots, but it cannot enlarge Pi's already-one-token request. The recorder
then received HTTP 200 with one completion ID and no logprobs and rejected the
collation. This locates the token-budget discrepancy upstream of the Harbor
extension, in Pi's own request/session budgeting; it does not establish why Pi
chose one token. No optimizer step passed. The disposable flag was removed from
the launch config after this terminal attempt; the committed config stays off.

A **CPU-only Daytona endpoint probe** then reproduced the same one-token cap
with the original task instruction, four tools, model alias, thinking format,
and the pinned Qwen tokenizer. The endpoint implements ordinary HTTP
`/tokenize` and streaming `/v1/chat/completions` in the sandbox; the completion
is scripted, so it is **not** a learned model or a teacher training trajectory.
Pi's initial cap was 1 at advertised contexts 2,048 and 4,096, even though
Qwen's true rendered prompt was 1,484 tokens; it became 256 when advertised
at 6,144. Inspection of the exact Pi 1.3.0 `pi-ai` package found a fixed
4,096-token client-side safety subtraction before Harbor's precise network
budget hook. Harbor now advertises an extra 4,096 virtual tokens to Pi and
adds the same offset to its compaction reserve, while `/tokenize` still clamps
to the **true** served context. A repeat disposable Daytona probe completed
all four Pi runs and observed cap 256 at true served contexts 2,048 and
4,096, the same 1,484-token prompt, and complete scripted literal evidence.
The sandbox was deleted. No new managed GPU optimizer step is implied.

A fresh managed GPU retry (`attempt-12`, sandbox verbose mode **off**) confirmed
the budget correction: four Daytona trials each reached two teacher model
callbacks with HTTP 200 and `max_completion_tokens: 256`; first-turn prompts
had 1,484 served IDs, and each first response included a tool call. The run
then failed closed before an optimizer step because the managed teacher turns
provided sampled IDs but **no per-token logprobs**, while Harbor's current
`PiModelEvidence` requires both. Host request/turn traces do not replace the
missing in-memory attributed trial. No automatic retry: the teacher-side
evidence contract needs a separate actor-aware fix that preserves strict
student behavior provenance without requiring unused teacher probabilities.

The follow-up Harbor collation fix accepts missing **teacher** logprobs only
when sampled prompt/response IDs, attributed assistant actions, and executed
tool observations are complete; student/revision turns still require aligned
per-token logprobs. The SkyRL hydration validator mirrors this distinction,
and SFT still retokenizes teacher actions with the student tokenizer. While
testing that fix, a CPU Daytona probe exposed occasional corruption of the
large single-line in-memory Pi evidence frame; Harbor now sends numbered,
bounded stdout chunks and rejects missing or duplicate chunks. A disposable
Daytona sandbox then ran **three** Pi teacher-trace trials against a CPU
scripted HTTP endpoint using Qwen tokenization, each with a tool call and a
second final response **without any teacher logprobs**. All three hydrated one
two-turn trajectory with its tool observation. The actor-aware recorder now
omits the `logprobs` and `top_logprobs` request fields for teacher-authored
turns and private teacher labels, so an upstream vLLM server need not compute
or ship unused scores. The SkyRL HTTP bridge only serializes sampled scores
when explicitly requested; its external-teacher validator accepts absent
scores but still rejects partial ones. A fourth disposable Daytona check used
a CPU endpoint that rejected any teacher request containing either logprob
field; Pi still hydrated both turns. All sandboxes were deleted. Unit tests
cover incomplete teacher/student evidence and two-turn SFT masking. This is a transport/contract regression, **not** a GPU optimizer
step, real teacher-quality result, or student evaluation.
  For remote packed task parquets,
  put the URI in the typed launch's `inputs.train_data` as well: the training
  driver stages and extracts that input on each node before SkyRL reads it.
  Putting an S3 URI only in the source recipe bypasses staging; PyArrow's
  native S3 client cannot use the worker's fsspec-injected CoreWeave endpoint.
- `PI_SMOKE_OUTPUT`: a durable trajectory URI co-located with the assigned GPU
  zone. Give each run an isolated prefix. Never use a workstation-local path.
  Each installed-agent attempt streams `engine/diagnostics.jsonl` snapshots
  to this prefix during execution (polled every two seconds, plus upload latency),
  with a heartbeat every 30 seconds and a final flush. The journal contains
  timestamped fixed phase names, machine-operation boundaries, and exception
  class names; never commands, stdout, model responses, endpoint URLs, or
  credential values. The latest published snapshot survives a hard worker
  death, although the last unflushed seconds may be lost. An upload failure
  fails the attempt rather than silently disabling diagnostics. A Daytona
  run timeout may delete the sandbox before container-owned logs can be
  downloaded; diagnose from the host journal rather than assuming the missing
  logs prove the agent never started.
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
