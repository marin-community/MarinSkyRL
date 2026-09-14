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
