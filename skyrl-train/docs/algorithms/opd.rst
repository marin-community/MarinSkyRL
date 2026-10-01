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

The production gate covers a Megatron Qwen3 policy with a separate, unquantized vLLM
teacher. It runs one optimizer step each night and requires finite teacher-advantage
metrics, positive teacher-scored and valid-token counts, a nonzero mean absolute
teacher advantage, and a positive raw gradient norm. The same gate can select an
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

Megatron objective adapters have CPU integration coverage. The recurring production gate
uses an unquantized local vLLM teacher. No quantized local-teacher configuration is
currently defined or production-gated. SGLang teacher scoring is rejected because it
cannot provide the required prompt logprobs.

Teacher objectives
------------------

``trainer.algorithm.distillation.objective=sampled_reverse_kl`` uses chosen-token
teacher scores to form detached policy advantages. At each valid training token,
the teacher advantage is the teacher coefficient times the route weight times
``teacher_logprob - old_policy_logprob``. The optional ``advantage_clip`` bounds
that log-probability difference symmetrically before either weight is applied.
Its default is ``null``. The driver consumes chosen-token evidence before sending
the resulting advantages to the learner.

With ``reward_mode=add``, teacher advantages follow environment-advantage
normalization and loop credit. With ``reward_mode=replace``, only positions with
valid teacher evidence are eligible for training, and teacher advantages supply
the policy credit. REPLACE requires ``advantage_estimator=uniform``, an
advantage-linear policy loss, no reward KL penalty, no advantage normalization,
no loop credit and no dynamic sampling. Configured KL and entropy loss rows still
train over the eligible positions.

``policy_loss_type=importance_sampling`` applies the ratio of current to old
policy probabilities to the teacher advantage without PPO clipping. Other
advantage-linear policy losses apply their own clipping or weighting rules.
The driver reports ``distillation/teacher_advantage_mean``,
``distillation/teacher_advantage_abs_mean``,
``distillation/teacher_advantage_clipped_fraction`` and
``distillation/valid_tokens``.

``sparse_forward_kl`` normalizes the teacher's retained top-K probabilities and
compares them with the student's full-distribution probabilities on that support.
``student_topk_policy_surrogate`` applies the clipped policy surrogate on the
rollout policy's selected support. These objectives carry top-K evidence to the
learner and form a separate teacher loss row. REPLACE sets their policy
advantages to zero. Both require TP=1, CP=1, sequence parallelism=1 and sample
packing disabled; ``advantage_clip`` applies only to chosen-token evidence.

All objective rows use denominators counted over the complete optimizer window
and data-parallel ranks. Route weights and the teacher coefficient weight the
numerator. They do not change token or sequence counts.

Routing and residency
---------------------

Routes map each admitted trajectory to one logical teacher and a loss weight. Multiple
remote replicas of a teacher are selected by pending token load and retryable failures
cool down an unhealthy endpoint. A fleet may combine remote teachers, pinned local
teachers, and local teachers that rotate through one drained residency slot. This is
the Multi-Teacher On-Policy Distillation (MOPD) mechanism: each admitted trajectory can
use its domain-specific teacher without coupling teacher placement to the trainer.
