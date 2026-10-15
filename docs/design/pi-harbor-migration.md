# Pi scenarios on the shared rollout engine

Harbor/Pi owns one live task trajectory and collates **one complete, attributed
trial payload in memory**: visited states, sampled responses and exact token
IDs/logprobs, separately attributed executed responses, tool observations,
private teacher labels and student-visible
feedback. Harbor validates the Pi payload; when task verification is enabled,
the grader returns a separate verdict. The rollout worker binds verification
status and lease identity to the payload and performs method-specific
ETL; it must not join model-request logs, Pi text output or separately stored
teacher records. Pi chooses the actor before sampling and executes the chosen
response. SkyRL config selects the typed Pi capability and the training
objective, not an extension filename.

The shared engine also collects model callbacks as `ModelTurn` records. Those
are **not** a lookup source for Pi ETL. Pi collates its attributed trial in its
own process; Harbor validates it and returns it in `AgentContext.metadata`.
The installed-agent adapter returns that same in-memory object to its task
session, which attaches it directly to the graded `PiRolloutData`. The worker
verifies its Pi-bound trial ID and policy step against the lease, then adds
an explicit graded or skipped verification status before scenario ETL.

## Scenario families

- **Demonstrate:** A teacher runs the task and the student imitates its
  action trace with SFT (`teacher_demo_pi_base`); explanatory text may be
  part of that trace, but the objective supervises actual teacher-authored
  tokens (Hsieh et al., 2023, *Distilling Step-by-Step*). Teacher
  **distribution** distillation instead requires teacher token probabilities
  for a specified context and a soft-target KL objective (Hinton et al.,
  2015, *Distilling the Knowledge in a Neural Network*). A saved trace alone
  does not provide those probabilities; it is not the Pi SFT scenario.
- **Evaluate:** Independent student trajectories receive post-trial scores
  from an AI judge for group-relative RL (`rlaif_pi_base`; Lee et al., 2023,
  *RLAIF*; Shao et al., 2024, *DeepSeekMath*). Judge provenance and scores
  remain distinct from executable verifier rewards. The current
  `verifier_grpo_base` uses **only** Harbor verifier scores; it is a working
  baseline, not an AI-judged GRPO smoke.
- **Correct:** At a student-visited state, the teacher labels a corrected
  action and the student SFT objective learns that label (DAgger turn mixture
  or prefix takeover; Ross et al., 2011, *DAgger*; Li et al., 2026,
  *Revisiting DAgger in the Era of LLM-Agents*). The correction must retain
  teacher authorship; it is not a sampled student response.
- **Critique and revise:** A teacher sends private critique of a student
  attempt; the student then revises before the verifier scores the revision
  (`critique_revision_pi_base`; Bai et al., 2022, *Constitutional AI*;
  Madaan et al., 2023, *Self-Refine*). Teacher feedback is not itself a
  verifier reward or a teacher-authored revision.

Only verifier GRPO and teacher trace SFT are wired for a launch. The other
families have constructed CPU evidence contracts but deliberately fail shared-
engine objective preflight. The short GPU smoke recipes and their required
artifacts are described in [Pi scenario smoke gates](../pi-scenario-smokes.md).

## Task ingestion boundary

A pure Pi launch reads executable Harbor tasks **directly** from
`data.train_data` with `data.kind: tasks`. Iris stages packed task archives to
local directories; `HarborTaskDataset` imports and lowers each task into a
`LoweredTaskSpec` before the shared worker runs it. Local Harbor directories,
TaskTrove selectors, and locally staged pre-lowered task parquets also converge
at that boundary. Pi does **not** require a second reference parquet with
`extra_info.nemotron_ultra.*`; `NemotronTaskDataset` keeps that join only for
actual mixed Nemotron corpora. Teacher labels and verifier outcomes are never
part of the source task definition.

The Harbor package importer is intentionally narrower than Harbor itself: it
supports single-stage Linux tasks with prebuilt images and a separate verifier
environment, but rejects Dockerfile builds, multi-stage tasks, shared verifiers,
healthchecks, MCP/skills directories, accelerator-type requirements, and
verifier collect hooks. Pre-lowered tasks have their own typed validation.
Installed Pi currently downloads apt/nvm/npm tools during setup, so its task
machine must allow outbound internet; the worker rejects `NetworkPolicy.DENY`
before starting a Pi trial. The CPU contract tests cover direct task ingestion
and this network preflight; neither proves CoreWeave sandbox egress.

## Current integration boundary

The pinned Harbor fork exposes typed `AgentConfig.pi` modes `literal`,
`teacher_trace`, `dagger`, and `critique_revision`. Harbor's in-container Pi
integration and its earlier model-backed Docker fixtures exercised the older
trial contracts. An opt-in **installed-package Docker trial** now validates the
new two-turn Pi payload with an executed bash tool and Harbor verifier; its
scripted HTTP tokens deliberately fail multi-turn GRPO prefix alignment. A
second opt-in Docker test runs installed Pi through `TaskRolloutWorker` with
an actual tiny CPU Qwen policy, separate Harbor verifier and single-turn
ETL. It compares the sampled prompt/response IDs and selected-token logprobs
at the model-client boundary with the graded training batch; no scripted
completions, IDs or logprobs enter this trial. Giving the tiny model an 8192
context window (rather than 4096) leaves enough room for Pi's prompt and
output reserve: Pi now requests up to 128 tokens and the policy emits a
multi-token answer (`#### 1`). This is not evidence for tool actions,
multi-turn prefix assembly, or realistic model quality. A separate Qwen2.5
multi-turn Docker probe below covers real tool execution. The installed worker
now accepts `teacher_trace` only with an independently bound teacher model,
revision, source kind, and (for external vLLM) authenticated endpoint. DAgger
and critique remain blocked; a distributed trainer update has not passed.
The older Harbor trajectory runner's teacher, critique, RLAIF and
replay adapters were not rebased into this branch.

`skyrl_train.rollouts.pi_scenario_etl` starts with a strict versioned envelope
and a pure `scenario.preset` → typed evidence → training-row transformation.
Its CPU fixtures test the format, student-tokenizer SFT conversion, distinct
verifier/judge rewards, revision masking, a `TrajectoryBatch` projection and
rejection of incomplete context. Harbor Pi now emits an attributed
`PiCollatedTrajectory`; the pinned Harbor dependency validates it and the
installed runtime passes it to `TaskRolloutWorker.training_batch` in memory.
CPU fixtures exercise the producer schema from the **installed pin** and the
worker's graded batch conversion; the opt-in Docker tests separately cover the
Harbor trial producer (scripted, negative-prefix test) and the installed-agent
shared-worker consumer (real sampled tokens), but not a trainer update. Multi-turn
GRPO assembles one whole-trial row, masking observed
context tokens; if successive exact served prefixes do not align it fails
closed rather than training on separate per-turn GRPO rows. Teacher trace
likewise assembles one physical SFT row per trial in the student tokenizer,
masking intervening tool/user context. A real teacher may sample tool-call JSON
with different whitespace than Pi's executed replay; SFT retokenizes the
validated executed rendering to preserve a continuous student-tokenizer
prefix while retaining the unmodified sampled teacher evidence in the trial.

## Training step and CPU evidence

The optimizer consumes token-level `rewards` (a list for every response). When
verification runs, its scores remain separately available in
`unshaped_rewards` and `verification_results`; skipped verification has a false
availability mask and no fabricated verifier score. Each Pi GRPO trial occupies one group member; GRPO
requires at least two physical trials per prompt, differing rewards for a
nonzero group advantage, exact-prefix assembled student logprobs, and the
regular clipped policy loss. Teacher trace uses one trial per prompt,
student-tokenizer SFT labels and the SFT policy loss with uniform advantages;
no sampled student behavior logprobs may be synthesized for teacher tokens.
The SFT loss does **not** weight tokens by the verifier score. Both paths train
whole trials rather than step-wise sessions.

`validate_pi_objective_config` checks scenario, installed Pi mode and selected
objective; teacher launch requires an independently pinned source and (for
managed teachers) GPU resources. The unfiltered preset disables task grading
and runs no in-training student evaluation. The verified preset retains the
task grader, excludes explicitly failed teacher trials from training via the
shared nonfatal fully-masked admission path, and refuses missing verdicts.
The teacher tokenizer and external endpoint's authenticated `/v1/models`
response are checked **before policy/rollout actor allocation**. CPU canaries in
`tests/cpu/tiny_training/test_pi_scenario_step.py` take hydrated trial fixtures
through ETL, group admission, production trainer tensor collation and advantage
calculation, production objective reduction and a tiny Qwen policy backward +
optimizer step. They discovered and fixed two previously hidden runtime gaps:
Hugging Face `BatchEncoding` from teacher retokenization and scalar rewards
incompatible with the trainer's per-token reward collation. A separate
opt-in Docker probe now runs Pi `teacher_trace` with Qwen2.5 teacher completions,
a tiny student tokenizer/model, a real bash observation, and production SFT
backward/optimizer step. Both a SkyRL-owned CPU inference engine and an
authenticated separate vLLM-wire HTTP fixture pass; neither is a deployed GPU
teacher. Distributed policy workers, GPU parity, checkpoint restart and a
learning-curve comparison remain untested. The current production preflight
admits `verifier_grpo_base`, `teacher_demo_pi_base`, and
`teacher_demo_pi_verified_base` with complete source bindings: other Pi ETL
presets have no wired trainer objective and fail at launch.
It checks selected mode and loss without silently rewriting training settings.
Managed teachers allocate their own pinned vLLM role with disjoint GPU actors;
the Iris role plan now reserves a separate whole-node teacher bundle (one extra
8-GPU node for the small smoke). External teachers are recorded as remote roles
and require no Iris GPU allocation. The trainer's validation worker
uses a separate literal Pi configuration bound to the **student** inference
engine. A separate opt-in tiny-policy Docker test passes the literal student
through the physical Harbor evaluation path under the teacher preset. The source
is **code-wired, not production-smoke validated**; do not
interpret a CPU Qwen endpoint fixture as vLLM or distributed parity.

Set `scenario.preset: teacher_demo_pi_base` for SFT without grading, or
`teacher_demo_pi_verified_base` for correct-only SFT. The preset also selects
`harbor.verifier_disable`; keep that decision out of individual data rows.
Unfiltered training uses `trainer.eval_interval: -1` and a separate student
evaluation run. The verified preset can schedule student-only Harbor evaluation
with `trainer.evaluation_runner: harbor`, a positive `trainer.eval_interval`,
and nonempty `data.val_data`. Under
`terminal_bench_config.installed_agent`, use `agent.pi.mode: teacher_trace`,
`agent.model_name: hosted_vllm/<teacher-alias>`, and a `teacher` binding with
`source`, `model` (the served alias), `model_path`, `revision`, `student_model`
(the exact configured policy path), and `max_context_tokens`. `external_vllm`
adds `endpoint: https://.../v1` and `auth: env:TEACHER_KEY`; it must expose an
authenticated `/v1/models` card with the same alias and context limit and the
selected model must have a locally resolvable tokenizer at the declared revision.
`skyrl_managed` instead adds `resources: {backend: vllm, num_engines: 1,
tensor_parallel_size: 1}` (optionally `gpu_memory_utilization`). No managed
teacher weights are synced to the policy; separate Ray GPU actors are allocated
and cleaned up after training. Do not point the teacher binding at the student
weights or use the training teacher as a validation actor.

| Scenario | Harness / Harbor | Trainer-side CPU result | Remaining design gate |
| --- | --- | --- | --- |
| Verifier GRPO | Single-turn tiny Qwen and two-turn real Qwen2.5 (bash tool, Docker, grade, exact prefix, sampled IDs/logprobs, trainer input) passed. | Full two-turn constructed envelope → shared worker → two-trial GRPO advantages, objective, tiny-Qwen backward and optimizer step passed. | Production vLLM tool-parser/template parity, distributed policy worker, physical GRPO group and checkpoint restart. |
| Teacher trace SFT | Real pinned Qwen2.5 teacher sampled a two-turn tool trajectory through Pi/Docker using both managed CPU and authenticated external vLLM-wire sources, distinct from the tiny student. | Shared worker retokenized executed teacher actions into one SFT row; production trainer collated, computed advantages, and stepped the tiny student's optimizer. Ungraded and verified-only admission have CPU contract tests. | Run ungraded SFT and a separate live student-only evaluation with GPU vLLM and distributed workers; checkpoint restart and weight provenance remain unverified. |

The migration must establish these *separate* evidence flows:

| Scenario | Required shared-engine evidence and objective |
| --- | --- |
| Verifier GRPO | Student-only Pi turns with Pi-collated sampled evidence; Harbor verifier grade remains the optimization reward. |
| Teacher trace imitation | Teacher-authored task turns with a canonical actor field, no student behavior logprobs on teacher turns. Student-tokenizer SFT rows, optional grader-gated admission, and separate student-only evaluation. |
| AI-judged GRPO | Student-only attributed turns, then a post-trial judge; record its structured score, request identity and provenance separately from Harbor's verifier grade. Group and optimize complete student trials only. |
| DAgger turn mixture / prefix takeover | Pi selects the actor **before** sampling an entire response. Its single collated trial contains exact visited context, private teacher label, selected executable response and observations with authorship. Retokenize teacher labels with the student tokenizer for SFT; do not treat them as behavior-policy tokens. Preserve bounded replay and its checkpoint state. |
| Critique → revision | Pi-collated student attempt and tool observations, private teacher request, only labeled feedback in the subsequent student context, then a distinct student revision. Mask all but revision tokens and use the separate Harbor verifier grade, not the teacher's opinion, as reward. |
| Teacher distribution distillation | A separate token-scoring path over exact student contexts and support, not a Pi plugin, trace or judge score. Use the existing distillation objective only when matching teacher distributions are actually available. |

For DAgger and teacher demonstrations, Harbor's pre-sampling controller
selects the actor and Pi produces the complete trial-local evidence. The
shared worker then projects **that same hydrated object** into training rows.
The producer and worker share the payload contract. `teacher_trace` now
requires a bound source distinct from the student; the worker spec accepts
independent injected managed engines or creates an authenticated external
vLLM-wire completion transport. The binding declares both the served alias
and the source model path/revision, and reused student engine objects are rejected.
A Pi recorder race discovered by the slow external fixture was fixed in Harbor:
final SSE bytes no longer precede the in-memory evidence frame without waiting
for the active request to finish. DAgger and critique still need their own
per-trial plans. The launch coordinator now creates an independent literal
student evaluation runner rather than using the teacher's inference backend.
The external endpoint's revision is **declared provenance**, not remote weight
attestation; `/models` checks its alias/context/auth but cannot prove the
remote weight hash. The managed engine is requested with the pinned model path
and revision, but no live GPU load/checkpoint-restart has yet been observed.
CPU probes
are not a production trainer run or GPU parity.

## Full-trajectory example coverage and smoke readiness

`tests/cpu/rollouts/test_pi_full_trajectory_examples.py` exercises all six
reviewed ETL presets against one complete file-reading task: a tool request,
tool observation, final answer, grade, and (where applicable) a private teacher
label, teacher-executed action, separate post-trial judge, or bounded critique
and revision. It asserts attribution, prefix/context masks, and reward-source
selection. These are constructed contract examples: the served IDs/logprobs and
chat-template fixture are **not sampled by a model**, and teacher/judge identities
are fixture data, not authenticated endpoints. All six pass in CPU ETL. Five
also pass the pinned Harbor producer-envelope validation, `TaskRolloutWorker`
assembly and production trainer collation, advantages, objective and backward;
verifier GRPO and teacher trace additionally run a tiny Qwen forward and
optimizer step on their complete worker-derived batches. RLAIF instead fails
closed at the shared worker: Pi's producer envelope has no post-trial judge
field and the worker currently sets `judge=None`. The separate opt-in Docker
test passes an installed Pi *single-turn, multi-token* real-policy trial, graded
and propagated through the shared worker.

An additional opt-in test (`RUN_HARBOR_PI_QWEN_TOOLS=1`) now runs an actual
pinned Qwen2.5-0.5B-Instruct CPU model through installed Pi and Docker: Qwen
samples a bash `cat` tool call, Pi executes it and observes a file value
absent from the task prompt, Qwen answers from that observation, Harbor grades
it, and the worker assembles one exact-prefix trial. The test checks sampled
IDs/logprobs at both turns against the supervised response tokens and checks
that the production trainer accepts the assembled input. The Qwen CPU bridge
parses Qwen-generated tool markup and normalizes OpenAI argument strings into
objects **only when rendering the next Qwen chat prompt**; without that
normalization the exact-prefix ETL correctly rejects the trial. It neither
invents actions nor changes sampled tokens. This 6-minute test still has one
trial, so it cannot claim nonzero GRPO group advantages or GPU parity. The
separate constructed two-trial CPU test establishes the GRPO optimizer path;
production vLLM tool-parser/template parity still needs smoke validation. Run
the opt-in probe with `RUN_HARBOR_PI_QWEN_TOOLS=1`; it requires Docker, about
1 GB of pinned Qwen weights from Hugging Face, and roughly six minutes on an
8-vCPU VM. It intentionally does not duplicate the one real trial to invent a
second GRPO group member. The same opt-in file has `managed-teacher` and
`external-teacher` cases: both run a real independent Qwen teacher against
a tiny student. The external case goes through a separately authenticated
vLLM-wire HTTP service and the production read-only external transport; a
wrong bearer token is rejected before sampling. Both retokenize Pi's executed
actions in the student's tokenizer into one SFT row and take a tiny-student
optimizer step. The HTTP fixture emulates vLLM's API using Qwen CPU weights;
it is **not** a running GPU vLLM server.

Teacher trace has tested CPU source routing for both types and launch now
selects a validated independent teacher and separate student-only evaluator.
The two runnable objectives have three reviewed Hydra presets: verifier GRPO,
unfiltered teacher SFT, and grader-gated teacher SFT. The Iris composer selects
one with `scenario.preset`.
RLAIF, both DAgger variants and critique→revision have ETL/producer contracts
but still fail objective preflight: none has complete per-trial source routing,
judge transport or replay checkpoint state in the shared worker. Online DPO
remains an older-runner prototype, not a Pi shared-engine objective. Do not
copy the legacy Hydra groups over: they use `scenario.kind` and the retired
runner, not the current hydrated trial. Distributed trainer updates,
checkpoint restart and GPU serving are still separate gates.

## Port and release gates

1. Harbor/Pi emits the versioned, self-contained trial envelope. Real Qwen
   Docker probes cover a tool followed by a final answer for both student and
   teacher modes; extend to executed multi-tool responses and a DAgger
   student/teacher transition, with no private label in the next student
   context and **no cross-log joins**.
2. Extend the connected shared installed-agent runtime to teacher/judge modes
   and adapt whole-task projections to each distinct objective; assert exact loss
   masks, student-tokenizer labels, group admission and replay checkpoint state.
3. Extend the Iris `scenario.preset` composer beyond verifier and teacher only
   as each objective acquires complete source routing and a CPU end-to-end test.
   Conflicting manual knobs and unsupported sources must fail before workers.
4. Exercise each method through the **installed pinned Harbor package** and the
   shared engine on CPU, then test trainer updates/checkpoint restart and GPU
   serving separately.

The original pre-shared-engine prototype is preserved at the local
`backup/pi-scenarios-before-shared-port` ref. It is a source of tests and
contracts, not an implementation to resurrect by mechanical rebase.
