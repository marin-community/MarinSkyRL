# Pi shared-engine smoke gates

These are **integration gates**, not model-quality runs. Use the typed Iris launcher
with the source recipes in `cloud/iris/configs/pi_verifier_grpo_smoke.yaml` and
`pi_teacher_trace_smoke.yaml`. Do not submit either recipe as a standalone Iris
job: the launcher supplies the pinned model source, staged local path, placement,
checkpoint paths, and immutable run/attempt identifiers. These recipes cover
**one of the five target methods** (teacher trace imitation under Demonstrate)
and a **non-target task-verifier GRPO baseline**. Verified-only teacher SFT is
an optional grading filter on teacher trace, not a sixth method. The
**Evaluate** target is *AI-judged* GRPO (`rlaif_pi_base`), not verifier GRPO;
its judge transport and physical group are not yet wired. The other targets
are distribution distillation, DAgger correction, and teacher critique →
student revision; see the [canonical five-method design](design/pi-harbor-migration.md).
`teacher_distribution_pi_base` also composes through `scenario.preset`, but
has **no committed GPU launch recipe yet**. It routes graded, student-only Pi
trials to SkyRL's existing admitted-group top-K teacher scorer and sparse
forward KL, not to Pi teacher-trace SFT. Only composition and a scripted-oracle
CPU worker/scoring/gradient contract have passed; do not count them as a live
vLLM teacher score or a trainer update. Its launch requires one pinned,
independent `teachers` entry with `topk_distribution`, exactly one route under
`teacher_routing`, a matching student/teacher tokenizer vocabulary, and separate
teacher resources. The typed parser compiles the teacher plan before composing
Hydra scenario defaults: the source recipe must explicitly declare matching
`trainer.algorithm.distillation` `objective: sparse_forward_kl`,
`reward_mode: replace`, `routing_plan` and `coefficient`, in addition to the
scenario selector. Conflicting values fail preflight. The Pi student preset
requests sampled-token logprobs with
`sampling_params.logprobs: 0`; positive logprobs (student-selected top-K) still
lack a Pi HTTP transport. The verifier GRPO preset now requests the same sampled
logprobs: Pi student evidence otherwise fails closed at collation.

The recipes compose through `scenario.preset`; the example recipes request
one or two 8×H100 nodes (the second node is the disjoint managed vLLM teacher
role). Match the
typed launch's node shape to the target cluster: `cw-us-east-08a` currently
has **4×GB200**, not 8×H100, so set `generator.num_inference_engines: 4`,
`trainer.placement.policy_num_gpus_per_node: 4`, and allocate 4×GB200 per
node there. Earlier isolated GPU teacher attempts failed; `attempt-13` later
completed the first optimizer step (recorded below).

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

The next isolated managed GPU attempt (`attempt-13`, Iris job
`/zack/pi-beta-teacher-gb200-smoke-20261016-m`) **succeeded** after pinning the
actor-aware Harbor collation and no-unused-teacher-logprobs request changes.
Iris reported both tasks succeeded, with no task failures or preemptions. The
shared rollout produced four durable training trajectory archives and the
trainer reached `global_step=1`: reported policy loss `2.93368923664093` and
gradient norm `62.8`. The step-1 checkpoint, including the model shards,
trainer state, and the co-located tokenizer, is at
`s3://marin-us-east-02a/MarinFold/pi-scenarios-shared-smoke-20261015/output/attempt-13/checkpoints/global_step_1/`.
Sandbox verbose diagnostics stayed off. This establishes a **real, single
teacher-trace SFT optimizer step**, not student improvement, separate student
evaluation, live DAgger or a physical verifier GRPO group. No W&B URL appeared
in the run logs; don't infer a logged W&B run from this smoke.

Student evaluation preparation (Oct 19): a separate four-GPU checkpoint-export job
`/zack/pi-beta-sft-export-20261019-n` completed and saved the step-1 HF
weights **and tokenizer** at
`s3://marin-us-east-02a/MarinFold/pi-scenarios-shared-smoke-20261015/output/eval-attempt-14/exports/global_step_1/policy/`
(manifest identity `sha256:82f44ae265ddbe8185d6a5343f65e7e9131710719a0b8b248642a315bcef087e`).
A separate eight-task `heldout-eval-v2` graded pack was staged under the same
regional bucket; none of its task IDs occurs in the SFT training pack.
The first student-only evaluation used an unsupported shared shell-verifier
mode and failed. The next step-1 evaluation (`eval-attempt-17-step1`) reported
Iris success and processed eight rows, **but all eight trials failed before
Pi sampled a token**: the evaluation-only runner supplied `global_step=None`
to the installed-agent policy-step lease. The paired step-0 job was canceled.
Do not report the all-zero aggregate as student accuracy or learning. The
follow-up uses the standard evaluation runner's step-0 lease, explicit
sampled-token logprobs in evaluation sampling, and a fail-closed scored-coverage
check. Fresh job IDs and output prefixes are required to retry both models.

With the step-0 policy lease and `eval_sampling_params.logprobs: 0` on SkyRL
`0721e2d6`, **both** independent evaluation jobs reached terminal success:
`/zack/pi-beta-student-eval-step1-20261019-r3` (`eval-attempt-20-step1`)
and `/zack/pi-beta-student-eval-step0-20261019-r3`
(`eval-attempt-21-step0`). Each sampled and graded eight distinct held-out
physical tasks, with verifier coverage `8/8`. Every task scored zero for
both models (`avg_score=0.0`, `pass_at_1=0.0`), and every response stopped at
the 256-token length limit. The saved SFT policy was loaded via its step-1 HF
export, but **no improvement was measured**. Per-task eval records were saved
under each attempt's `exports/dumped_evals/eval_only/`. A more discriminative
eval needs enough answer budget and tasks on which at least one model produces
a verified pass; the equal all-zero scores are only a valid integration result.

The first physical verifier-GRPO attempt used the disjoint eight-task
`input/verifier-grpo-v1/tasks.parquet` pack, two student samples per prompt,
sampled-token logprobs and a 512-token per-turn cap. Iris job
`/zack/pi-beta-verifier-grpo-gb200-20261019-o` (`grpo-attempt-22`) failed
before the first optimizer step: one trial returned corrupted numbered
in-memory evidence (invalid base64), while other trials failed to receive a
score because separate verification timed out (`VerifierTimeoutError` /
`VerificationNotCompletedError`). A `ValidationError` appeared in the
subsequent cleanup trace for the corrupted-frame trial; it is **not** evidence
that a separate grader rejected an artifact. The shared worker refused to
substitute missing trial evidence, so **no complete physical GRPO group or
update has been validated**. Do not retry the GPU job unchanged. One disposable Daytona/Qwen3-tokenizer probe
completed a scripted 361-token student response with 361 sampled logprobs;
this does not reproduce the intermittent corrupt frame. A six-trial follow-up
failed during the Pi/NVM setup before any student trial ran. Reproduce the
stdout transport and separate-verifier timeouts in disposable CPU sandboxes
first, without retaining raw Pi stdout or treating diagnostic logs as
trajectory evidence.

The **next** Pi smoke recipes now set an 8,192-token true request/serving
window and a 2,048-token per-turn output cap in
`cloud/iris/configs/pi_{teacher_trace,verifier_grpo}_smoke.yaml`. Their derived
trainer and agent input allowance is 6,144 tokens; the previous 2,048-token
window left only 564 slots after the observed 1,484-token first prompt. This
change has **not** reached a successful evaluation or GPU optimizer job.
A fresh 8,192/2,048 step-1 eval (`eval-attempt-23-step1`, Iris job
`/zack/pi-beta-student-eval-step1-20261019-r4`) failed before graded results:
Harbor rejected a malformed numbered in-memory Pi evidence chunk. Do not
count it as an evaluation. A separate disposable Daytona/CPU check under the
same nominal limits completed one scripted 1,401-token student response with
sampled logprobs and valid capture; it does not establish multi-turn GPU
transport reliability. The matched step-0 GPU eval was **not** launched after
this failure. Manual VM Docker runs now exercise the installed Pi agent through
the shared worker with real CPU-sampled Qwen tokens, including a pretrained
Qwen2.5-0.5B tool call, executed tool, separate verifier and ETL. Those
opt-in tests use the SkyRL CPU engine with a vLLM-compatible HTTP bridge,
**not** vLLM's CPU runtime or the CoreWeave/Daytona GPU path; do not use them
to debug vLLM behavior. A separate manual probe then ran **real vLLM 0.32.0
on CPU** (pinned `vllm/vllm-openai-cpu` image, independent Qwen3-0.6B weights
at the student revision, true 8,192 context and 2,048 per-turn cap). Pi ran
inside the pinned Python task image as an installed agent, talking directly
to that vLLM's `/tokenize` and `/v1/chat/completions` endpoints—not the SkyRL
CPU inference bridge. Three separate disposable Docker trials completed two
turns each. They sampled 1,458 + 446, 670 + 189, and 1,601 + 470 tokens
respectively, with aligned student logprobs; the second trial executed two
tool observations for two calls, and the third executed one for one. No raw
prompt, response, command or stdout was saved. The reusable manual harness is
`skyrl-train/tests/cpu/rollouts/manual_pi_vllm_docker.py`: point `--task-uri`
at a **one-row** Harbor task archive parquet, `--image` at the pinned task
image, `--vllm-base` at a real vLLM server reachable from Docker bridge, and
`--model` at the served Qwen name. It validates and prints only fixed-schema
sandbox diagnostics and in-memory counts. The server used pinned CPU image
`vllm/vllm-openai-cpu@sha256:827ef38187f9b52f2bb838695ef815ec5b65f4bf3e4c061b002b2b0a19cb0701`,
student weights at `c1899de289a04d12100db370d81485cdf75e47ca`, 8,192
`max_model_len`, `float32` (this VM's EPYC CPU lacks fast bf16), and
`--enable-auto-tool-choice --tool-call-parser hermes`. This restores a real
CPU-vLLM/Pi debugging loop, but vLLM 0.32.0 is **not** Marin's pinned GPU
fork, and local Docker is **not** Daytona's stdout transport; passing these
trials does not explain the intermittent eight-task GPU frame corruption.
Harbor's pinned producer now labels malformed evidence by fixed, content-free frame shape
(missing separator, invalid header, empty or oversized fragment) to locate
the transport fault on a fresh trial without retaining stdout or leaking task
content; the earlier failed GPU attempt cannot be classified retroactively. The next
one-task step-1 GPU diagnostic (`eval-attempt-25-step1-debug`, Iris job
`/zack/pi-beta-student-eval-step1-20261019-r5`) **succeeded** with those new
limits and the diagnostic Harbor pin. Its single student Pi trial produced
2,048 sampled response tokens, stopped for length, received an independently
attributed task-verifier grade (`score=0`), and was counted in `num_scored=1/1`
under `exports/dumped_evals/eval_only/`. This is a live long-response
transport/grading integration pass, **not** model improvement, a solved task,
a full eight-task evaluation, or evidence that the intermittent concurrent
frame corruption and separate-verifier timeouts are fixed. Do not reuse the old 256-/512-token launch YAMLs or any spent
attempt IDs/output prefixes; ensure fresh launch configs inherit the preset
budget without a smaller ad-hoc override. The larger budget does not repair
corrupt in-memory evidence, separate-verifier timeouts, or all-zero GRPO
advantage.

Next smoke gates, in order:

1. The saved step-1 student was independently loaded and evaluated against the
   step-0 baseline, with separate per-task records; both scored 0/8 under the
   256-token cap. Repeat with a meaningful answer budget and tasks that produce
   valid, non-degenerate outcomes before drawing learning conclusions.
2. Diagnose the in-memory stdout transport and separate-verifier timeouts
   locally, then run a physical verifier-GRPO **baseline** group with
   two independently sampled Pi trials per prompt, exact per-turn sampled
   logprobs, task-verifier grades, nonzero group advantage and a checkpointed
   GPU update. Never copy one trial to manufacture a group member.
3. Before GPU distribution distillation, test an installed Pi student task
   against a real independently pinned vLLM teacher oracle on CPU/small GPU,
   with exact token alignment, masked tool context and an actual shared-trainer
   teacher-loss update. The earlier Qwen2.5-0.5B teacher cannot score Qwen3
   token IDs: their vocabulary fingerprints differ (`8f16caac…` versus
   `3689317f…`). Pinned `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
   matches the Qwen3-0.6B student's vocabulary fingerprint and is an
   independent teacher-weight source for the next **integration** test; no
   teacher-quality claim follows from this check. Then do a one-step,
   disjoint-teacher GPU smoke with a fresh attempt ID and output prefix; do
   not infer quality improvement.
4. Only after a separate post-trial judge returns identity, score and provenance
   without overwriting verifier evidence, run a physical AI-judged GRPO group.
   DAgger and critique→revision each need their own connected objective/source
   and restart gates before GPU smoke tests.
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
