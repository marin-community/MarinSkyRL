Online Policy Distillation
==========================

Online policy distillation (OPD) adds teacher evidence to the policy objective
without coupling the trainers to a particular teacher deployment. Synchronous
training overlaps teacher scoring with the policy forward pass. Fully asynchronous
training submits scoring after rollout-group admission and waits for evidence only
when assembling a learner batch. Both regimes use the same routing plan, oracle
contract, replica pool, and objective payload.

Supported configurations
------------------------

The production gate covers an FSDP2 Qwen3 policy with a separate, unquantized vLLM
teacher. It runs one optimizer step each night and requires finite distillation loss,
teacher-scored tokens, and a positive raw gradient norm. The same gate can select an
OpenAI-compatible endpoint fixture for transport acceptance.

Local teachers currently require the vLLM backend and ``pinned`` or ``rotating``
placement. Remote teachers require an HTTP(S) ``/v1`` completions base endpoint that
supports token-ID prompts, prompt logprobs, ``echo``, and
``return_tokens_as_token_ids``. Generic chat-completions APIs do not satisfy this
contract. Remote replicas declare their concurrency and may use bearer credentials
resolved from environment, file, or versioned GCP secret references.

Vocabulary-level objectives require identical token-ID semantics. Local teacher
vocabularies are fingerprinted from their tokenizer before GPU allocation. Remote
teachers declare the same SHA-256 vocabulary fingerprint, and every response must echo
the exact requested token IDs. A matching tokenizer family name is not sufficient.

FSDP/DeepSpeed and Megatron objective adapters have CPU integration coverage. Only
FSDP2 with an unquantized local vLLM teacher is covered by the recurring production
model gate. No quantized local-teacher configuration is currently defined or
production-gated. SGLang teacher scoring is rejected because it cannot provide the
required prompt logprobs. Other policy/backend combinations should be treated as
experimental until they gain targeted GPU coverage.

Routing and residency
---------------------

Routes map each admitted trajectory to one logical teacher and a loss weight. Multiple
remote replicas of a teacher are selected by pending token load and retryable failures
cool down an unhealthy endpoint. A fleet may combine remote teachers, pinned local
teachers, and local teachers that rotate through one drained residency slot. This is
the Multi-Teacher On-Policy Distillation (MOPD) mechanism: each admitted trajectory can
use its domain-specific teacher without coupling teacher placement to the trainer.

Sequential curricula
--------------------

Sequential OPD runs one ordinary typed Iris job per stage, so only that stage's
teacher fleet needs to be reachable. The workflow is independent of the training
entrypoint: a stage job may use synchronous or fully asynchronous training, and the
next stage receives the immutable Hugging Face policy export returned by the same
Iris job protocol.

Build each stage's typed job JSON with ``marinskyrl iris build-request``. A strict
curriculum YAML then references those jobs and records the information that changes
the result::

  version: 1
  curriculum_id: math-then-code
  output_root: s3://bucket/experiments/math-then-code
  stages:
    - id: math
      domain: math
      job_spec: math-job.json
      teachers: [math]
      data_mixture:
        - {domain: math, source_identity: "math-data@revision"}
      token_budget: 1000000
      sampling: {temperature: 0.7, max_generate_length: 2048, n_samples_per_prompt: 4}
      state: {optimizer: reset, scheduler: reset, rng: reset}
      retention_evaluations:
        - {domain: math, source_identity: "math-eval@revision"}
    - id: code
      domain: code
      job_spec: code-job.json
      teachers: [code]
      data_mixture:
        - {domain: code, source_identity: "code-data@revision"}
        - {domain: math, source_identity: "math-rehearsal@revision"}
      token_budget: 2000000
      sampling: {temperature: 0.5, max_generate_length: 4096, n_samples_per_prompt: 2}
      state: {optimizer: continue, scheduler: continue, rng: continue}
      retention_evaluations:
        - {domain: math, source_identity: "math-eval@revision"}
        - {domain: code, source_identity: "code-eval@revision"}

Run or resume the manifest with::

  marinskyrl iris run-opd-curriculum --manifest curriculum.yaml

The launcher validates that recorded teachers and data identities match each typed
job, and that every stage evaluates its current domain plus all earlier domains.
Sampling values become Hydra overrides rather than duplicated configuration. The
token budget is also operational: both trainers stop after the first completed
optimizer batch that reaches the cumulative teacher-scored-token budget.

State continuation is deliberately atomic because every supported checkpoint backend
loads optimizer, scheduler, and RNG state together. A stage must mark all three
``continue`` or all three ``reset``. Continue loads the prior full checkpoint but
starts a new stage-local step count and data cursor; reset initializes from only the
prior policy export. In-stage retries retain their token count, while a new stage
resets it before applying its own budget.

The output root contains ``curriculum.json`` plus one immutable JSON record per stage.
Each record includes the resolved input policy, optional full checkpoint, teacher and
data manifests, sampling and state decisions, retention evaluations, Iris job ID,
resolved job, terminal manifest, and exported policy. These intermediate artifacts
are sufficient to resume the remaining suffix or select an earlier retention
tradeoff; changing any input produces a different manifest digest and is rejected
against existing state.
