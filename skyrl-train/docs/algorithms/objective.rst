Policy objective architecture
=============================

The RL objective at a glance
----------------------------

The policy worker minimizes one loss per optimizer step:

.. math::

   L = \mathrm{loss\_scale}\left[\mathrm{policy} + \beta\,\mathrm{KL}
       - \eta\,\mathrm{entropy} + \mathrm{teacher}\right].

Each bracketed term starts as one value per response token and is averaged
over the whole optimizer step, across every microbatch and data-parallel rank.
The code calls these terms rows (``ObjectiveRows``). This averaging does not depend on how the batch is
split; the covariance losses can still depend on microbatch statistics.
``loss_scale`` undoes how Megatron and DDP combine the microbatches and ranks.

What it is made of, and where it lives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Paths are relative to ``skyrl-train/skyrl_train/``.

.. list-table::
   :header-rows: 1
   :widths: 14 30 24 32

   * - Part
     - What it does
     - Module
     - Main objects
   * - Policy loss
     - One value per token from the current policy, the old policy, the rollout policy and the advantages. It never averages and never corrects
     - ``objective/losses.py``
     - ``PolicyLossInputs``, ``TokenLoss``; the losses ``regular``, ``dual_clip``, ``importance_sampling``, ``behavior_clip``, ``gspo``, ``cispo``, ``sapo``, ``clip_cov``, ``kl_cov``, ``sft``
   * - Averaging
     - Counts each term's denominators once per optimizer step, then turns per-token values into the step value. Data weights set the denominator; correction and route weights only scale the numerator
     - ``objective/reduction.py``
     - ``WeightCounts``, ``StepCounts``, ``step_counts``, ``reduce_to_step``
   * - Off-policy correction
     - Per-token weights from the old-policy / rollout-policy ratio. They multiply the policy term only
     - ``objective/correction.py``
     - ``compute_correction``, ``CorrectionResult``; presets ``tis``, ``icepop``, ``seq_mask_tis``, ``outlier_mask``
   * - Teacher signal
     - A sampled teacher log-probability becomes an advantage. A top-K teacher distribution becomes the teacher term
     - ``objective/teacher.py``
     - ``teacher_advantages``, ``topk_teacher_loss``
   * - KL and entropy
     - Regularizers against the reference model, and for exploration
     - ``utils/policy_math.py``
     - ``differentiable_approx_kl``; ``KLEstimator`` (declared in ``config/objective_spec.py``)
   * - Composition
     - Builds one micro-batch's inputs, computes the loss above, and reports each term as its step value
     - ``objective/objective.py``
     - ``ObjectiveMicroBatch``, ``ObjectiveRows``, ``PolicyObjective``, ``compute_policy_objective``, ``megatron_loss_scale``
   * - Declarations and checks
     - Torch-free, so the launcher runs them before submission; the same checks run again at start-up
     - ``config/objective_spec.py``
     - ``RatioAnchor``, ``LossSpec``, ``BUILTIN_LOSS_SPECS``, ``LossReduction``, ``TopKLossParams``, ``topk_loss_params``, ``rollout_logprobs_required``, ``validate_objective``; ``OffPolicyCorrection``, ``off_policy_correction``
   * - Non-finite steps
     - On Megatron, skips an optimizer step whose gradients are NaN or infinite on any rank; fails when the consecutive-skip allowance is exhausted
     - ``distributed/step_policy.py``
     - ``NonfiniteStepPolicy``, ``OptimizerStepResult``, ``nonfinite_step_policy``
   * - Names
     - Maps each ``policy_loss_type`` to its function
     - ``utils/algorithm_registry.py``
     - ``PolicyLossType`` (from ``marinskyrl.runtime_options``), ``register_policy_loss``

Where each part runs
~~~~~~~~~~~~~~~~~~~~

1. **At launch.** ``validate_objective`` rejects contradictory or ignored
   settings, both before the job is submitted and when the driver starts.
   Workers and rollout consumers compute teacher-loss parameters and logprob
   requirements and correction rules from the current config through Torch-free
   helpers. Named correction presets are immutable and cached by name. Custom loss
   declarations travel with their functions through the Ray registry.

2. **On the driver, once per training batch** (``trainer.py``):

   #. The forward pass gives the old policy's log probabilities.
   #. REPLACE mode masks the tokens that have no teacher evidence.
   #. ``compute_correction`` computes the policy correction weights.
   #. The advantage estimator computes the environment advantages.
   #. ``teacher_advantages`` adds the sampled teacher signal to the advantages.

3. **On the worker, once per optimizer step:**

   #. ``step_counts`` (one all-reduce).
   #. ``compute_policy_objective`` for each microbatch.
   #. The optimizer step, which applies, skips or fails.

How to choose each part
~~~~~~~~~~~~~~~~~~~~~~~

All keys are under ``trainer.algorithm``, unless shown otherwise.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Part
     - Keys
   * - Policy loss and averaging
     - ``policy_loss_type``, ``loss_reduction``, ``advantage_estimator``
   * - Correction
     - ``off_policy_correction`` (a preset name, or ``none``)
   * - Teacher
     - ``distillation.objective``, ``distillation.reward_mode`` (``add`` or ``replace``), ``distillation.coefficient``, ``distillation.advantage_clip``
   * - KL and entropy
     - ``use_kl_loss``, ``kl_loss_coef``, ``kl_estimator_type``, ``use_entropy_loss``, ``entropy_loss_coef``
   * - Non-finite steps
     - ``trainer.policy.max_consecutive_nonfinite_steps``
   * - A whole algorithm
     - ``config_groups.algorithm_recipe``: ``grpo``, ``dapo``, ``dr_grpo``, ``gspo``, ``cispo``, ``opd``, ``mopd``

For the full rules, see the detailed sections below,
the :doc:`objective_guide`, and :doc:`opd` for teacher deployment and routing.


Averaging over an optimizer step
--------------------------------

``loss_mask`` records which response tokens train. Environment, tool, template
and padding positions contribute neither loss nor counts. Policy data weights
multiply that mask by ``think_token_weight`` on THINK-tagged tokens. KL and
entropy use the raw mask. Top-K teacher data weights use the mask intersected
with teacher-valid positions.

For token loss :math:`\ell_{it}`, data weight :math:`d_{it}` and optional
numerator weight :math:`w_{it}`, the numerator is
:math:`\sum_{it}d_{it}w_{it}\ell_{it}`. Importance corrections and teacher route
weights affect this numerator. They do not enter the counts. In particular,
masking a token with a correction does not renormalize the surviving tokens.

Let :math:`T=\sum_{it}d_{it}`, :math:`n_i=\sum_t d_{it}`,
:math:`B=\sum_i\mathbf{1}[n_i>0]`, and :math:`L_{\max}` be the configured
total input and generation length,
``generator.max_input_length + generator.sampling_params.max_generate_length``.
``StepCounts`` computes counts over every microbatch
in one optimizer step and sums them across data-parallel ranks in one packed
all-reduce. Context-parallel replicas are excluded from that count reduction.
Counts are recomputed for each accumulation window and training epoch.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - ``trainer.algorithm.loss_reduction``
     - Policy and top-K teacher reduction
   * - ``token_mean``
     - :math:`\sum_{it}d_{it}w_{it}\ell_{it}/\max(T,1)`.
   * - ``sequence_mean``
     - :math:`\sum_{i:n_i>0}(\sum_t d_{it}w_{it}\ell_{it}/n_i)/\max(B,1)`.
   * - ``seq_mean_token_sum_norm``
     - :math:`\sum_{it}d_{it}w_{it}\ell_{it}/(\max(B,1)L_{\max})`.
   * - ``seq_mean_token_sum_norm_global``
     - :math:`\sum_{it}d_{it}w_{it}\ell_{it}/(\max(B_{A\ne0},1)L_{\max})`, where :math:`B_{A\ne0}` counts responses with nonzero data-weighted advantages. Top-K teacher terms cannot use this mode.

KL always uses ``sequence_mean`` with raw-mask counts. Entropy always uses
``token_mean`` with raw-mask counts. Empty responses contribute zero; denominator
counts are clamped to at least one. For ``sequence_mean``, each nonempty response
is divided by its true weighted mass :math:`n_i`, including masses below one
when ``think_token_weight < 1``. Invalid positions are zeroed before
nonlinear operations, so a masked NaN does not contaminate a valid loss.

For two responses with 10 and 1,000 trainable tokens and unit data weights,
``token_mean`` gives the short response 10/1,010 of the total token weight and
the long response 1,000/1,010. ``sequence_mean`` gives each response half the
weight. Splitting the responses into different microbatches or data-parallel
ranks preserves those weights. The covariance losses ``clip_cov`` and
``kl_cov`` select tokens using microbatch statistics; their selection itself
can change when the partition changes.

Driver and worker order
-----------------------

The driver admits rollout groups and attaches teacher evidence before the old
policy forward. It obtains old-policy log probabilities, then intersects the
training mask with teacher validity for REPLACE. Mismatch diagnostics therefore
retain their original input view. The driver computes environment advantages,
optionally normalizes them, adds loop credit, and finally applies chosen-token
teacher credit. Chosen-token evidence is consumed on the driver; top-K evidence
travels to the learner.

Each worker buffers an accumulation window, computes its step counts, and runs
the current-policy forward. It builds finite masked inputs, evaluates the
per-token losses, reduces the terms against those counts, and calls backward.
The optimizer then applies or skips the synchronized step.

Teacher credit and teacher terms
--------------------------------

For chosen-token evidence, with route weight :math:`u_i`, coefficient
:math:`\gamma_T` and optional symmetric clip :math:`c`, the driver constructs

.. math::

   A^T_{it}=\gamma_T u_i\,\operatorname{clip}
      (\log\pi_T(a_{it})-\log\pi_{\rm old}(a_{it}),-c,c).

All quantities in this advantage are detached. With ``advantage_clip: null``
the gap is unbounded. The coefficient and route weight multiply the gap once,
after clipping. ``reward_mode: add`` uses :math:`A=A^{\rm env}+A^T` after
environment normalization and loop credit. ``reward_mode: replace`` uses
:math:`A=A^T` and trains only teacher-valid positions.

Top-K evidence carries selected token IDs, teacher log probabilities, validity
and route weights. The worker evaluates the current student on that support
and adds a teacher term. ADD keeps the environment policy term; REPLACE gives it
zero advantages and zero policy loss. Route weights and the distillation
coefficient multiply the teacher numerator; the denominator counts valid data.

REPLACE requires ``advantage_estimator: uniform`` and an advantage-linear
policy loss such as ``importance_sampling``. Here advantage-linear means that
zero advantages give zero policy loss and gradient. It also requires
``use_kl_in_reward: false``, ``advantage_batch_normalize: false``, zero loop
advantage penalty and ``dynamic_sampling.type: null``. ``use_kl_loss`` and
``use_entropy_loss`` remain independent regularizers; set both false for a pure
teacher objective. Deployment and routing are described in :doc:`opd`.

Validation and failure stages
-----------------------------

``validate_launch_config`` checks a composed objective before job submission.
``validate_cfg`` repeats the contract at runtime startup after resolving custom
registrations. Consumers compute objective settings from the current config;
custom loss declarations travel with their callables through the Ray registry.
Teacher startup checks tokenizer identity and capabilities;
writer admission checks actual rollout evidence. After admission, the driver
checks teacher response identities before learner-batch assembly. An optimizer-step
failure is a separate runtime event.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Stage
     - Check or failure
     - Remedy
   * - Launch/startup
     - Unknown loss or reduction; a custom loss lacks its specification.
     - Select a registered loss and one of the four reductions, or register the custom loss with a ``LossSpec`` before startup.
   * - Launch/startup
     - GSPO requires sequence-level credit and ``sequence_mean``.
     - Use ``sequence_mean`` and disable token-level loop or sampled-teacher credit, or choose a token-level loss.
   * - Launch/startup
     - ``think_token_weight != 1`` requires token reward channels.
     - Set ``enable_token_reward_channel: true`` or use weight 1.
   * - Launch/startup
     - Sampled teacher credit with ``sft``; incompatible REPLACE settings.
     - Use an advantage-consuming loss; for REPLACE apply the requirements above.
   * - Launch/startup
     - Top-K teacher geometry or reduction is unsupported.
     - Set ``use_sample_packing: false`` and policy sequence parallelism, TP and CP to 1; use one of the first three reductions.
   * - Launch/startup
     - ``advantage_clip`` with top-K evidence.
     - Use chosen-token ``sampled_reverse_kl`` or set ``advantage_clip: null``.
   * - Launch/startup
     - Invalid student-top-K surrogate settings.
     - Use ``token_mean``, ``0 <= eps_clip_low < 1``, ``eps_clip_high >= 0`` and ``clip_ratio_c > 1``.
   * - Launch/startup
     - Invalid non-finite step limit.
     - Set ``trainer.policy.max_consecutive_nonfinite_steps`` to null or an integer at least 1.
   * - Teacher startup
     - Teacher/student tokenizer identity or evidence capability mismatch.
     - Use identical vocabularies and tokenizer fingerprints, and a teacher backend that serves the requested evidence.
   * - Writer admission
     - Missing or malformed required rollout log probabilities.
     - Enable the required generator evidence and preserve its token alignment through rollout processing.
   * - Driver, after admission
     - Teacher evidence response identities differ from the admitted trajectories.
     - Preserve trajectory IDs and response order through teacher scoring and learner-batch assembly.
   * - Optimizer step
     - Non-finite loss/gradients or an exhausted skip allowance.
     - Inspect the affected batch, objective and gradient diagnostics; correct the numerical cause before resuming.

Non-finite optimizer steps
--------------------------

Megatron with ``trainer.policy.max_consecutive_nonfinite_steps: 3`` permits
three non-finite skips between applied updates and raises on the next. Null
raises immediately. Explicit null values in a launch document override declared
defaults, so this null limit reaches the trainer. A successfully applied update
resets the streak; finite
gradient-threshold skips leave it unchanged. The base worker does not implement
this skip allowance. The Megatron skip decision is synchronized
across the affected ranks. A skipped non-finite step preserves parameters,
optimizer moments and the learning-rate scheduler. A finite gradient norm above
a configured optimizer threshold follows that threshold's own semantics.

Backward and reporting scales
-----------------------------

For data-parallel size :math:`D` and :math:`M` microbatches per accumulation
window, the base worker uses ``loss_scale = D`` to cancel gradient averaging.
Megatron uses ``loss_scale = M * D`` because its schedule also divides by
:math:`M`. The objective applies this scale once, after composing the terms.

Both paths use ``report_scale = M * D``. The existing metric collection averages
microbatch/rank contributions, so this scale makes reported terms equal the
full optimizer-step values. Backward and reporting scales have separate
arguments; changing metric aggregation must not change the gradient scale.
They are internal worker information, not user configuration fields.

Unclipped importance sampling
-----------------------------

``importance_sampling`` uses the ratio :math:`r=\pi_\theta/\pi_{\mathrm{old}}`,
the same ratio as ``regular``, and minimizes :math:`-rA` without PPO clipping.
They give the same update with one optimizer step per batch, when current and
old policies coincide during gradient computation. They can differ when a batch
is split into several optimizer steps and clipping becomes active; accumulating
microbatches into one step does not create that difference.

TIS (``off_policy_correction: tis``) is independent: it weights by the detached
ratio :math:`\pi_{\mathrm{old}}/\mu`, capped at 2, and can combine with this loss.
`Tinker's importance_sampling loss <https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/>`_
uses the sampler's probabilities as its denominator; this loss uses
:math:`\pi_{\mathrm{old}}`.

Adding a policy loss
--------------------

Register a function with ``register_policy_loss(name, spec=...)`` and return
``TokenLoss(values, metrics)`` where ``values`` has the same response-token
shape as its input log probabilities. Do not average within that function.
``LossSpec`` declares the ratio anchor, sequence-level credit requirement,
dependence on microbatch statistics and zero-advantage behavior. These
declarations determine which configurations are valid before training starts. See
:doc:`custom_algorithms` for the registration example.

PPO/TIS score centering
----------------------

``trainer.algorithm.score_centering_topk`` defaults to zero (disabled). A
positive width adds an expected-score correction to regular PPO with exactly
one token-level TIS truncation rule. Set ``generator.sampling_params.logprobs``
to the same width. This requires local vLLM and unpacked Megatron with tensor,
context and sequence parallel size one; distillation is unsupported.

The correction uses current, stored pre-update and sampled-behavior log
probabilities for identical candidate IDs. It includes directional PPO clipping
and the stored/behavior TIS cap, and is reduced with the policy's whole-step
counts. The sampled action's TIS weight must not multiply it again. Outside the
captured head, stored and behavior probabilities are approximated as copies of
the current policy scaled to preserve each tail mass. This approximation does
not guarantee the true omitted gradient; tail-mass metrics report its coverage.
The three tail-mass means weight eligible tokens by the policy's THINK-token
weight, using global optimizer-step counts before averaging across steps.
Equivalent microbatch and data-parallel partitions preserve these means.
Stored probabilities remain the anchor throughout training of the batch;
current chosen and candidate scores share a differentiable normalizer.

For cap 1.05 and width 32:

.. code-block:: yaml

   trainer:
     algorithm:
       policy_loss_type: regular
       off_policy_correction: custom
       off_policy_correction_rules:
         - {kind: token, action: truncate, high: 1.05}
       score_centering_topk: 32
   generator:
     sampling_params:
       logprobs: 32

.. _objective-kl-estimator:

KL estimator
------------

The reference regularizer targets reverse KL,
:math:`D_{\rm KL}(p\Vert q)=\mathbb{E}_{a\sim p}[\log p(a)-\log q(a)]`,
where :math:`p` is the current policy and :math:`q` the frozen reference. For a
sampled action let :math:`r=q(a)/p(a)`, :math:`\delta=\log p(a)-\log q(a)`
and :math:`g=\nabla_\theta\log p(a)`. The following expectations assume fresh
samples from :math:`p`, a fixed context, common support and inactive clamps.
Autograd treats sampled actions as fixed.

.. list-table::
   :header-rows: 1
   :widths: 12 18 22 18 30

   * - Estimator
     - Per-token value
     - Expected value
     - Per-token gradient
     - Expected gradient
   * - ``k1``
     - :math:`-\log r=\delta`
     - Reverse KL; individual values may be negative.
     - :math:`g`
     - Zero.
   * - ``k2``
     - :math:`\tfrac12(\log r)^2`
     - Nonnegative, biased approximation to reverse KL.
     - :math:`\delta g`
     - :math:`\nabla D_{\rm KL}(p\Vert q)`.
   * - ``k3``
     - :math:`r-1-\log r`
     - Reverse KL; nonnegative per token.
     - :math:`(1-r)g`
     - :math:`\nabla D_{\rm KL}(q\Vert p)` (forward KL).
   * - ``abs``
     - :math:`|\delta|`
     - Absolute log-ratio penalty, not a KL estimate.
     - :math:`\operatorname{sign}(\delta)g`
     - Gradient of the sampled absolute penalty; no KL-gradient identity.
   * - ``k3_unbiased_gradient``
     - :math:`\operatorname{sg}(k3)+k2-\operatorname{sg}(k2)`
     - The k3 value.
     - :math:`\delta g`
     - :math:`\nabla D_{\rm KL}(p\Vert q)`.

For k3 autograd, :math:`\mathbb{E}_p[g]=0` and
:math:`\mathbb{E}_p[rg]=\mathbb{E}_q[g]`, so the expected gradient is
:math:`-\mathbb{E}_q[g]=\nabla D_{\rm KL}(q\Vert p)`.
A reverse-KL derivative must also differentiate the sampling distribution:

.. math::

   \nabla\mathbb{E}_p[k3]
   =\mathbb{E}_p[(1-r)g+k3\,g]
   =\mathbb{E}_p[\delta g]
   =\nabla D_{\rm KL}(p\Vert q).

``k3_unbiased_gradient`` uses a detached k3 value with the k2 gradient to supply
that coefficient. It is the default for the KL loss and for value-only metrics
and reward penalties. The latter run without gradients and receive the k3
value. `verl's KL implementation
<https://verl.readthedocs.io/en/latest/_modules/verl/trainer/ppo/core_algos.html#kl_penalty>`_
calls the value/gradient construction ``k3+``; the SkyRL configuration name is
``k3_unbiased_gradient``.

Configure the estimator explicitly when selecting a different convention:

.. code-block:: yaml

   trainer:
     algorithm:
       use_kl_loss: true
       kl_loss_coef: 0.001
       kl_estimator_type: k3_unbiased_gradient

``kl_estimator_type`` accepts exactly the five table entries. Use ``k3`` when
its direct autograd convention is intentional, ``k2`` when its biased reported
value is acceptable, ``k1`` for the signed sampled log ratio, and ``abs`` for an
absolute penalty. ``use_kl_estimator_k3`` and ``use_abs_kl`` are unsupported
configuration keys; select the estimator by name. Unknown names and unsupported
keys fail before job submission and at runtime validation.

Both k3 variants clamp :math:`\log(q/p)` to [-20, 20] and the reported k3 value
to [-10, 10]. For ``k3_unbiased_gradient`` the k2 backward term uses that clamped
log ratio but has no value clamp: its gradient remains active when only the k3
output saturates, and is zero beyond the log-ratio clamp. Explicit ``k2`` uses
the unclamped log difference. These numerical clamps limit the unbiased-gradient
statement, as do stale/off-policy samples and context-distribution changes.
The estimator name does not promise an unbiased full-training-objective gradient
under those conditions. The implementation is in ``utils/policy_math.py``.
