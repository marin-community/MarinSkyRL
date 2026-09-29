Policy objective architecture
=============================

The policy worker minimizes a sum of globally normalized objective rows:

.. math::

   L = \mathrm{loss\_scale}\left[
       R_{\rm policy}(\ell_{\rm policy})
       + \beta R_{\rm KL}(\ell_{\rm KL})
       - \eta R_{\rm entropy}(H)
       + R_{\rm teacher}(\ell_{\rm teacher})\right].

``trainer.algorithm.kl_loss_coef`` sets :math:`\beta` and
``entropy_loss_coef`` sets :math:`\eta`; ``use_kl_loss`` and
``use_entropy_loss`` enable their respective terms. A top-K teacher adds the
teacher row. Chosen-token teacher evidence enters the policy advantages.

The modules below are relative to ``skyrl-train/skyrl_train/``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``config/objective_spec.py``
     - Torch-free loss contracts, reduction names, resolved configuration and launch checks.
   * - ``objective/losses.py``
     - Policy formulas returning ``TokenLoss`` with one value per response token.
   * - ``objective/reduction.py``
     - Data weights, optimizer-step counts and reduction of each microbatch's contribution.
   * - ``objective/teacher.py``
     - Detached chosen-token advantages and differentiable top-K teacher losses.
   * - ``objective/objective.py``
     - Row composition, coefficients, backward scaling and reported values.
   * - ``utils/policy_math.py``
     - KL estimators and advantage normalization.
   * - ``distributed/megatron/nonfinite_steps.py``
     - Synchronized optimizer decisions for non-finite gradients.

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
     - :math:`\sum_{it}d_{it}w_{it}\ell_{it}/(\max(B_{A\ne0},1)L_{\max})`, where :math:`B_{A\ne0}` counts policy rows with nonzero data-weighted advantages. Top-K teacher rows cannot use this mode.

KL always uses ``sequence_mean`` with raw-mask counts. Entropy always uses
``token_mean`` with raw-mask counts. Empty rows contribute zero; denominator
counts are clamped to at least one. Invalid positions are zeroed before
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
per-token losses, reduces the rows against those counts, and calls backward.
The optimizer then applies or skips the synchronized step.

Teacher credit and teacher rows
-------------------------------

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
and adds a teacher row. ADD keeps the environment policy row; REPLACE gives it
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
registrations. Resolved contracts contain only primitive configuration values
for worker transport. Teacher startup checks tokenizer identity and capabilities;
writer admission checks actual rollout evidence. After admission, the driver
checks teacher row identities before learner-batch assembly. An optimizer-step
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
     - Teacher evidence row identities differ from the admitted trajectories.
     - Preserve trajectory IDs and row order through teacher scoring and learner-batch assembly.
   * - Optimizer step
     - Non-finite loss/gradients or an exhausted skip allowance.
     - Inspect the affected batch, objective and gradient diagnostics; correct the numerical cause before resuming.

Non-finite optimizer steps
--------------------------

Megatron with ``trainer.policy.max_consecutive_nonfinite_steps: 3`` permits
three non-finite skips between applied updates and raises on the next. Null
raises immediately. A successfully applied update resets the streak; finite
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
:math:`M`. The objective applies this scale once, after composing the rows.

Both paths use ``report_scale = M * D``. The existing metric collection averages
microbatch/rank contributions, so this scale makes reported rows equal the
full optimizer-step values. Backward and reporting scales have separate
arguments; changing metric aggregation must not change the gradient scale.
They are internal worker information, not user configuration fields.

Adding a policy loss
--------------------

Register a function with ``register_policy_loss(name, spec)`` and return
``TokenLoss(values, metrics)`` where ``values`` has the same response-token
shape as its input log probabilities. Do not average within that function.
``LossSpec`` declares the ratio anchor, sequence-level credit requirement,
row-local computation and zero-advantage behavior. These declarations determine
which configurations are valid before training starts. See
:doc:`custom_algorithms` for the registration example.
