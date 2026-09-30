Choosing an RL objective
========================

Choose the advantage estimator, policy loss, averaging and rollout correction
independently. A recipe supplies a compatible starting point; explicit settings
in the launch YAML override it. Teacher objectives add either policy advantages
or a separate loss row. :doc:`objective` describes the optimizer-step reduction
and distributed execution contract.

In the formulas below, :math:`p` is the current policy, :math:`o` the policy
scored before the update, :math:`b` the rollout policy, :math:`q` a teacher,
:math:`A` a detached advantage, and :math:`\operatorname{sg}` stops gradients.
Let :math:`r=p/o` and
:math:`C(r,A)=\max(-rA,-\operatorname{clip}(r,1-\epsilon_l,1+\epsilon_h)A)`.
Ratios computed by ``safe_exp_delta`` clamp the log difference to [-20, 20].
The formulas describe eligible tokens; ``loss_mask`` excludes every other
position before reduction. Source links name the implementation of each option.

Policy losses
-------------

Set these keys under ``trainer.algorithm``. The `policy loss source`_ defines
all ten options, including the covariance selections and clipping metrics.

``regular``
~~~~~~~~~~~

:math:`\ell=C(r,A)`. Use PPO clipping for token advantages. Choose
``importance_sampling`` when the intended update must have no PPO clipping.

.. code-block:: yaml

   policy_loss_type: regular
   eps_clip_low: 0.2
   eps_clip_high: 0.2

``dual_clip``
~~~~~~~~~~~~~

:math:`\ell=\min(C(r,A),-cA)` for :math:`A<0`, and :math:`C(r,A)` otherwise.
Use it to bound negative-advantage penalties at large ratios; use ``regular``
when that extra bound is not intended.

.. code-block:: yaml

   policy_loss_type: dual_clip
   eps_clip_low: 0.2
   eps_clip_high: 0.2
   clip_ratio_c: 3.0

``importance_sampling``
~~~~~~~~~~~~~~~~~~~~~~~

:math:`\ell=-rA`. Use for unclipped advantage-weighted updates, including
chosen-token OPD. Avoid it when large ratios require a clipped policy surrogate.

.. code-block:: yaml

   policy_loss_type: importance_sampling

``behavior_clip``
~~~~~~~~~~~~~~~~~

:math:`\ell=C(p/b,A)`, with the dual bound :math:`\min(\ell,-cA)` for
:math:`A<0`. Use to clip directly against the policy that generated the token.
It requires rollout log probabilities and cannot use a separate OLD-anchored
correction. Choose ``regular`` plus a correction to keep those ratios separate.

.. code-block:: yaml

   policy_loss_type: behavior_clip
   eps_clip_low: 0.2
   eps_clip_high: 0.2
   clip_ratio_c: 3.0
   off_policy_correction: none

``gspo``
~~~~~~~~

:math:`\ell_{it}=C(\exp(\min(\log p_{it}-\operatorname{sg}(\log p_{it})+
\operatorname{sg}(\operatorname{mean}_{t}\log r_{it}),10)),A_i)`.
This GSPO-token construction uses a sequence ratio in the forward pass and
individual token gradients. Use sequence-level rewards and ``sequence_mean``;
avoid sampled-teacher or loop-token credit. See `Group Sequence Policy
Optimization <https://arxiv.org/abs/2507.18071>`_.

.. code-block:: yaml

   policy_loss_type: gspo
   loss_reduction: sequence_mean
   eps_clip_low: 0.0003
   eps_clip_high: 0.0004

``cispo``
~~~~~~~~~

:math:`\ell=-\operatorname{sg}(\operatorname{clip}(r,1-\epsilon_l,1+\epsilon_h))A\log p`.
Use detached clipped importance weights while retaining token log-likelihood
gradients. This is a different gradient convention from PPO's pessimistic
selection; choose ``regular`` for that selection. The recipe's bounds are
[0, 6]. See `MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning
Attention <https://arxiv.org/abs/2506.13585>`_.

.. code-block:: yaml

   policy_loss_type: cispo
   cispo:
     cispo_eps_clip_low: 1.0
     cispo_eps_clip_high: 5.0

``sapo``
~~~~~~~~

:math:`\ell=-A(4/\tau)\sigma(\tau(r-1))`, with :math:`\tau=\tau_+` for
positive advantages and :math:`\tau_-` otherwise. Use a smooth sigmoid gate
instead of hard PPO clipping when that is the intended surrogate. Its sigmoid
can saturate; it does not reproduce PPO clipping. See the `policy loss source`_.

.. code-block:: yaml

   policy_loss_type: sapo
   sapo:
     tau_pos: 1.0
     tau_neg: 1.05

``clip_cov``
~~~~~~~~~~~~

:math:`\ell=(1-m)C(r,A)`, where :math:`m` selects a random subset of eligible
covariance-band tokens not already PPO-clipped. Use for experiments suppressing
that subset; avoid it when the objective must be invariant to microbatch
partitioning. Covariances use the current microbatch. See the `policy loss source`_.

.. code-block:: yaml

   policy_loss_type: clip_cov
   clip_cov:
     clip_ratio: 0.0002
     clip_cov_lb: 1.0
     clip_cov_ub: 5.0

``kl_cov``
~~~~~~~~~~

:math:`\ell=-rA+\lambda m|\log r|`, where :math:`m` selects the largest
advantage/log-probability covariances in the microbatch. Use for covariance-based
penalty experiments. Its penalty can train with zero advantages, so REPLACE
rejects it; it also depends on microbatch partitioning. See the `policy loss source`_.

.. code-block:: yaml

   policy_loss_type: kl_cov
   kl_cov:
     kl_cov_frac: 0.2
     ppo_kl_coef: 1.0

``sft``
~~~~~~~

:math:`\ell=-\log p`. Use for unweighted likelihood training on eligible
response tokens, including selected best-of-N responses. It ignores advantages,
so it cannot consume sampled-teacher credit or satisfy REPLACE.

.. code-block:: yaml

   policy_loss_type: sft
   advantage_estimator: uniform
   off_policy_correction: none

Averaging modes
---------------

Let :math:`N=\sum d w\ell`, :math:`T=\sum d`, :math:`B` count nonempty
responses and :math:`L_{\max}` be the total configured input and generation length
(``generator.max_input_length + generator.sampling_params.max_generate_length``). Numerator weights
:math:`w` include corrections and teacher route weights. Counts include only
data weights :math:`d`. The `reduction source`_ implements all four modes;
:doc:`objective` gives the full formulas, empty-row behavior and 10/1,000-token
example. Set the indicated key under ``trainer.algorithm``.

.. list-table::
   :header-rows: 1
   :widths: 36 25 39

   * - Exact configuration
     - Formula
     - Use and limitation
   * - ``loss_reduction: token_mean``
     - :math:`N/\max(T,1)`
     - Equal weight per data-weighted token. Longer responses contribute more total weight; use sequence averaging if that is unwanted.
   * - ``loss_reduction: sequence_mean``
     - Mean of each nonempty response's data-weighted token mean.
     - Equal response weight; required by GSPO. Short responses give each token more weight.
   * - ``loss_reduction: seq_mean_token_sum_norm``
     - :math:`N/(\max(B,1)L_{\max})`
     - Fixed length normalization for Dr.GRPO. Changing the configured length changes gradient magnitude.
   * - ``loss_reduction: seq_mean_token_sum_norm_global``
     - :math:`N/(\max(B_{A\ne0},1)L_{\max})`
     - Normalize by responses with nonzero data-weighted policy advantage. Avoid for top-K teacher rows, where validation rejects it.

Rollout corrections
-------------------

Corrections multiply the policy numerator by detached :math:`w`, using
:math:`\rho=o/b` on trainable tokens. They leave KL, entropy, teacher rows and
denominators unchanged. Only active OLD-anchored policy rows accept correction
rules. ``behavior_clip`` already uses :math:`p/b`; ``sft`` has no ratio.
The `correction source`_ and `preset source`_ specify these exact weights.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - ``trainer.algorithm.off_policy_correction``
     - Formula
     - Use and limitation
   * - ``tis``
     - :math:`w=\min(\rho,2)`
     - Limit large importance weights. Small ratios still contribute; use a mask preset to exclude outliers.
   * - ``icepop``
     - :math:`w=\rho\mathbf{1}[0.5\le\rho\le5]`
     - Retain in-range ratios and exclude token outliers. At ratio 1.5 the weight is 1.5. Avoid if discarded-token frequency is excessive.
   * - ``seq_mask_tis``
     - :math:`w_{it}=\mathbf{1}[0.99\le\exp(\operatorname{mean}_t\log\rho_{it})\le1.01]\min(\rho_{it},2)`
     - Reject sequences outside a narrow geometric-ratio band, then apply TIS. This can discard most stale data.
   * - ``outlier_mask``
     - :math:`w_{it}=\mathbf{1}[\forall t:10^{-4}\le\rho_{it}\le100]`
     - Exclude a whole response with any extreme trainable token. Retained responses have unit weight, so this does not importance-weight them.
   * - ``none``
     - :math:`w=1`
     - Explicitly accept uncorrected policy updates. Useful as a measured baseline; asynchronous OLD-anchored training requires an explicit choice.
   * - ``null``
     - :math:`w=1`
     - Unspecified correction for synchronous training. Rejected for asynchronous OLD-anchored updates; use ``none`` to make that choice explicit.
   * - ``custom``
     - Product of rule weights, with at most one ratio-bearing truncate rule.
     - Express measured token/sequence thresholds. Rules require inspection of retained data and mean weights; they are not automatically unbiased.

For ICEPOP, a ratio above 5 contributes to both ``policy/correction/truncated_fraction``
and ``policy/correction/masked_fraction``: its ratio cap and outlier mask both apply.
The final weight is zero. These fractions describe each rule, so they are not disjoint.

For example, this custom correction combines a geometric sequence mask and a
token ratio cap. These keys are under ``trainer.algorithm``:

.. code-block:: yaml

   off_policy_correction: custom
   off_policy_correction_rules:
     - kind: sequence
       aggregate: geometric
       action: mask
       low: 0.9
       high: 1.1
     - kind: token
       action: truncate
       high: 2.0

``kind: token`` uses :math:`\rho_{it}`. ``kind: sequence`` uses
``aggregate: geometric`` (exponentiated mean log ratio), ``product``
(exponentiated sum log ratio clamped to [-20, 20]), or ``extreme_token``
(all eligible token ratios must satisfy the mask). ``action: mask`` supplies an
inclusive interval indicator; either bound may be null. ``action: truncate``
supplies :math:`\min(\rho,h)` for its selected aggregate, requires ``high`` and
allows no low bound. Bounds must be positive and finite, with low <= high.
``extreme_token`` supports masks only. At most one truncate rule is allowed;
multiple masks multiply. Presets accept no custom rule list.

Advantage estimators and group filtering
----------------------------------------

Set ``trainer.algorithm.advantage_estimator`` to the listed value. Here
:math:`R_i` is a response reward sum, :math:`G` its prompt group, :math:`V` a
critic prediction and :math:`\operatorname{whiten}` masked centering and variance
normalization. The `advantage source`_ defines exact masking and singleton behavior.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Exact configuration
     - Formula
     - Use and limitation
   * - ``advantage_estimator: uniform``
     - :math:`A_{it}=1`
     - Unit credit for SFT; required for REPLACE, which then supplies teacher credit. Does not learn from environment reward by itself.
   * - ``advantage_estimator: reward``
     - :math:`A_{it}=\sum_{t':\mathrm{eligible}}r_{it'}` on eligible tokens.
     - Direct outcome reward without a group baseline. Useful for reward-weighted experiments; reward offsets directly affect the update.
   * - ``advantage_estimator: grpo``
     - :math:`A_i=(R_i-\bar R_G)/(s_G+10^{-6})`; omit the denominator with ``grpo_norm_by_std: false``.
     - Prompt-relative credit with a complete physical group. Use multiple responses for a baseline; use ``reward`` for direct single-response credit. See `DeepSeekMath`_.
   * - ``advantage_estimator: rloo``
     - :math:`A_i=R_i-\operatorname{mean}_{j\in G,j\ne i}R_j`.
     - Leave-one-out outcome baseline over a complete group. A singleton gets zero. Use ``rloo_n`` when infrastructure failures must be excluded. See `Back to Basics`_.
   * - ``advantage_estimator: rloo_n``
     - RLOO over baseline-eligible responses; excluded rows and undersized groups get zero.
     - Exclude infrastructure failures while retaining agent failures in the baseline. Set ``group_advantage_min_size: 2``; do not use for groups intended to train from one eligible response.
   * - ``advantage_estimator: rloo_n_pbs``
     - :math:`A=A^{\rm RLOO-N}+\mathrm{token\_level\_shaping}\times\mathrm{response\_mask}`.
     - Add the existing potential-based token shaping channel to RLOO-N. Without that channel it equals RLOO-N; avoid for strictly sequence-level GSPO credit.
   * - ``advantage_estimator: reinforce++``
     - :math:`A_t=\operatorname{whiten}(\sum_{k\ge t}\gamma^{k-t}r_k)`.
     - Critic-free discounted returns with masked whitening. Set ``gamma: 1.0`` for undiscounted returns; use a group estimator for a prompt-relative baseline. See `REINFORCE++`_.
   * - ``advantage_estimator: gae``
     - :math:`A_t=\operatorname{whiten}(\sum_{l\ge0}(\gamma\lambda)^l(r_{t+l}+\gamma V_{t+l+1}-V_{t+l}))`.
     - Temporal credit with a critic; requires ``trainer.critic.model.path``. Set ``gamma: 1.0`` and ``lambd: 1.0`` for undiscounted full returns. Avoid when no critic is provisioned.

For RLOO-N and RLOO-N-PBS, ``group_advantage_min_size`` must be at least 2 and
no larger than ``generator.n_samples_per_prompt``. An explicit integer is required; null is rejected. ``rloo_n_filter_zero_reward_groups: true`` assigns
zero credit to constant-reward groups. This estimator-local rule is separate
from writer admission's dynamic sampling.

The success-rate filter keeps final trial groups with
:math:`\bar R<c` and population reward standard deviation
:math:`s_R>s_{\min}` (singletons bypass the spread check). For binary raw
rewards :math:`\bar R` is success rate. For shaped/nonbinary rewards it is a
reward mean. Use to train on informative groups; avoid filtering when every
collected group must contribute, and do not enable it for teacher REPLACE.
The `group filter source`_ defines the writer-admission check.

.. code-block:: yaml

   dynamic_sampling:
     type: filter
     informative_on: unshaped
     min_reward_std: 0.0
     max_mean_reward: 0.9

``informative_on: shaped`` uses final shaped rewards; ``unshaped`` requires
raw outcome evidence. Null ``max_mean_reward`` disables the mean cutoff.

Teacher objectives
------------------

Configure the following keys under ``trainer.algorithm.distillation`` and
supply teachers and routing as in the worked setups. ``coefficient`` and route
weights multiply each teacher contribution once. The `teacher loss source`_
and `teacher configuration source`_ define these objectives. See
`MOPD section 3.2.1 <https://arxiv.org/html/2606.30406v1#S3.SS2.SSS1>`_
for the sampled single-token reverse-KL policy-gradient estimator.

``sampled_reverse_kl``
~~~~~~~~~~~~~~~~~~~~~~

:math:`A^T=\gamma_T u\operatorname{clip}(\log q-\log o,-c,c)`.
Use chosen-token teacher feedback through an advantage-consuming policy loss.
``advantage_clip`` bounds a teacher's log-probability gap before weighting;
null leaves it unbounded. This is a sampled policy-gradient signal, not an
exact full-vocabulary KL calculation. Avoid ``sft`` and GSPO for this credit.

.. code-block:: yaml

   objective: sampled_reverse_kl
   reward_mode: replace
   coefficient: 1.0
   routing_plan: opd
   advantage_clip: 2.0

``sparse_forward_kl``
~~~~~~~~~~~~~~~~~~~~~

:math:`\ell=\sum_{k\in S}\bar q_k(\log\bar q_k-\log p_k)`, where
:math:`\bar q_k=q_k/\sum_{j\in S}q_j`. Use teacher top-K evidence to match its
conditional distribution on retained support. Student probabilities keep their
full-distribution normalization. With partial support this is not full KL;
use tail-binned objectives when the aggregate tail should participate.
Optional ``entry_clip`` caps each summand above before summation.

.. code-block:: yaml

   objective: sparse_forward_kl
   reward_mode: replace
   coefficient: 1.0
   routing_plan: opd
   entry_clip: null

``sparse_reverse_kl``
~~~~~~~~~~~~~~~~~~~~~

:math:`\ell=\sum_{k\in S\cup\{\mathrm{tail}\}}p_k\log(p_k/q_k)`.
Use selected teacher tokens plus one aggregate tail bin to penalize student
mass relative to teacher mass. The selected probabilities are not renormalized;
the tail is one minus their sum. It equals full reverse KL only at full support.
For partial support, the teacher tail is computed from the available top-K
logprobs using float64 ``logsumexp`` and ``log1mexp``, with a probability floor
of ``1e-12``. This bounds the tail's negative log probability at approximately
27.63 even when float32 retained mass rounds to one. The floor regularizes the
tail bin; full support uses an exact zero tail.

.. code-block:: yaml

   objective: sparse_reverse_kl
   reward_mode: replace
   coefficient: 1.0
   routing_plan: opd

``sparse_jsd``
~~~~~~~~~~~~~~

:math:`\ell=\alpha\mathrm{KL}(q\Vert m)+(1-\alpha)\mathrm{KL}(p\Vert m)`,
with :math:`m=\alpha q+(1-\alpha)p` on selected tokens plus the tail bin.
Use a mixture divergence when neither teacher-to-student direction alone is
intended. Partial support still aggregates all omitted tokens into one bin.
``jsd_beta`` must be strictly between 0 and 1.
It uses the same float64, floored teacher tail as ``sparse_reverse_kl``.

.. code-block:: yaml

   objective: sparse_jsd
   reward_mode: replace
   coefficient: 1.0
   routing_plan: opd
   jsd_beta: 0.5

``student_topk_policy_surrogate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\ell=\sum_{k\in S_b}\operatorname{DualClip}(p_k/b_k,
-\operatorname{sg}[(\log p_k-\log q_k)\operatorname{softmax}_{S_b}(\log p)_k])`.
Use rollout-selected support and teacher scores on that same support for a
clipped distillation surrogate. The softmax weights and advantages are detached;
this is not an exact full KL estimator. It requires ``token_mean`` and rollout
``logprobs`` equal to every teacher's ``top_k``. The surrogate alone preserves
configured sampling and logprob handling, including ``top_p: 0.99`` in the
native Open-MOPD schedule. Neutral, processed behavior-logprob sampling applies
when an active policy row uses a scalar rollout ratio: a rollout-anchored loss
or a nonempty off-policy correction. Admission checks the surrogate's behavior
top-K fields independently of that scalar-ratio requirement. The runner must
preserve exact sampled completion tokens to capture that evidence.

.. code-block:: yaml

   objective: student_topk_policy_surrogate
   reward_mode: replace
   coefficient: 1.0
   routing_plan: opd

All top-K objectives require policy TP=CP=sequence parallelism=1 and sample
packing disabled. Use ``topk_distribution`` teacher evidence for sparse
divergences and ``student_selected_topk`` for the surrogate. ``advantage_clip``
applies only to chosen-token evidence. Teacher/student token-ID vocabularies
must match exactly. Deployment, tokenizer fingerprints and endpoint requirements
are in :doc:`opd`.

Recipes and complete setups
---------------------------

Recipes are composed through ``config_groups.algorithm_recipe`` in a launch
source YAML, or ``+algorithm_recipe=NAME`` in a direct Hydra invocation.
The `recipe source`_ contains the same paper references as the table below.
These recipes configure objectives; model, data, placement and generation
settings must also be supplied.

.. list-table::
   :header-rows: 1
   :widths: 16 43 41

   * - Recipe name
     - Objective settings and formula
     - Use and limitation
   * - ``grpo``
     - Group-standardized advantage, :math:`C(r,A)`, ``sequence_mean``, clip 0.2/0.2, KL coefficient 0.04.
     - Equal response weighting with reference regularization; short responses give larger per-token weight. `DeepSeekMath`_.
   * - ``dapo``
     - Group-standardized advantage, :math:`C(r,A)`, ``token_mean``, clip 0.2/0.28, no KL, nonzero reward-spread filtering.
     - Token-weighted clipped updates with informative groups. Avoid when every collected group must train. `DAPO`_.
   * - ``dr_grpo``
     - Mean-centered unstandardized group advantage, :math:`C(r,A)`, fixed-length normalization, clip 0.2/0.2, no KL.
     - Avoid reward-standard-deviation and response-length normalization; configured maximum length controls magnitude. `Understanding R1-Zero-Like Training`_.
   * - ``gspo``
     - GSPO-token sequence ratio, ``sequence_mean``, clip 0.0003/0.0004, explicit KL-off recipe choice.
     - Sequence-level credit with a narrow ratio interval. Avoid token-varying credit. `Group Sequence Policy Optimization <https://arxiv.org/abs/2507.18071>`_.
   * - ``cispo``
     - Detached ratio-weighted :math:`-A\log p`, ``token_mean``, ratio bounds [0,6], no KL, nonzero reward-spread filtering.
     - Retain log-likelihood gradients with clipped weights; does not implement PPO's pessimistic selection. `MiniMax-M1 <https://arxiv.org/abs/2506.13585>`_.
   * - ``opd``
     - Uniform then teacher REPLACE advantage, :math:`-rA^T`, ``token_mean``, no KL.
     - Pure chosen-token teacher training. Requires teacher definitions, routing and coefficient. Sampled estimator: `MOPD section 3.2.1 <https://arxiv.org/html/2606.30406v1#S3.SS2.SSS1>`_.
   * - ``mopd``
     - Uniform then routed teacher REPLACE advantage, :math:`-rA^T`, ``sequence_mean``, ``advantage_clip: 5.0``, no KL.
     - Give each response equal teacher-loss weight across domains. Domain frequency and route weights still affect the mixture. `MOPD <https://arxiv.org/abs/2606.30406>`_.

The GSPO paper's displayed objective has no KL term. Section 2 says the term
is omitted for brevity, and the paper reports no KL coefficient. This recipe
therefore sets ``use_kl_loss: false`` and ``use_kl_in_reward: false`` rather
than inheriting the base default. A user who wants reference KL sets it and
its coefficient explicitly.

The GRPO recipe uses DeepSeekMath's sequence averaging and KL coefficient 0.04.
The existing ``examples/gsm8k/run_gsm8k.sh`` uses the base ``token_mean`` and
KL coefficient 0.001. Those are distinct experiment choices. The DAPO recipe
uses the paper's regular clipped objective; the existing DAPO examples use
verl's ``dual_clip`` variant with ``clip_ratio_c: 10``. These example settings
are preserved. The recipe configures algorithm fields only: overlong filtering
(``generator.apply_overlong_filtering``) and soft length penalties
(``generator.trajectory_reward_shaping.overlong``) need explicit generator
settings. DAPO section 4.1 uses a 16,384-token expected maximum and an additional
4,096-token soft penalty interval; set generation limits and the shaping interval
together for the intended experiment.

For each of GRPO, DAPO, Dr.GRPO, GSPO and CISPO, start from this complete launch
source and change only ``algorithm_recipe`` to the table's name. Use the
launch configuration composer with the resulting source as the ``skyrl`` subtree
of your infrastructure's launch document. Set its model input and cluster
allocation to match two nodes with one GPU each. The
parquet files must contain prompts and rewards for ``gsm8k``. Dataset creation is
covered by :doc:`../datasets/dataset-preparation`.

.. code-block:: yaml

   entrypoint: standard
   config_groups:
     algorithm_recipe: grpo
   context_budget:
     request_window_tokens: 2048
     max_new_tokens_per_turn: 512
     max_turns: 1
   environment:
     env_class: gsm8k
   data:
     kind: parquet
     train_data: [/data/gsm8k/train.parquet]
     val_data: [/data/gsm8k/validation.parquet]
   trainer:
     strategy: megatron
     train_batch_size: 8
     policy_mini_batch_size: 8
     micro_train_batch_size_per_gpu: 1
     micro_forward_batch_size_per_gpu: 1
     max_steps: 10
     placement:
       colocate_all: false
       policy_num_nodes: 1
       policy_num_gpus_per_node: 1
       ref_num_nodes: 1
       ref_num_gpus_per_node: 1
     algorithm:
       off_policy_correction: none
   generator:
     backend: vllm
     run_engines_locally: true
     num_inference_engines: 1
     inference_engine_tensor_parallel_size: 1
     n_samples_per_prompt: 4

The enclosing `launch document schema`_ supplies ``run``, ``runtime``, ``iris``,
``ray``, ``artifacts`` and staged ``inputs``. Save that document as ``launch.yaml``;
``load_launch_config`` composes its source ``skyrl`` subtree and validates it
before submission. From the repository root, the launch command is:

.. code-block:: bash

   uv run --frozen python -m cloud.iris.launch iris launch --config launch.yaml

Asynchronous updates
~~~~~~~~~~~~~~~~~~~~

Keep the model/data/placement setup above and set these training fields. Separate
policy and rollout placement is required for positive staleness.

.. code-block:: yaml

   trainer:
     rollout_buffer:
       max_staleness_steps: 2
     algorithm:
       policy_loss_type: behavior_clip
       off_policy_correction: none

This clips :math:`p/b` directly. To clip :math:`p/o` and separately weight by
:math:`o/b`, choose ``policy_loss_type: regular`` with
``off_policy_correction: tis`` (or another preset). To measure an uncorrected
OLD-anchored baseline, use ``regular`` with explicit ``none``. The launcher
requests chosen-token rollout log probabilities when the objective needs them;
writer admission rejects missing, misaligned or non-finite required evidence.
Null correction is invalid for asynchronous OLD-anchored policy training.

OPD alone and added to RL
~~~~~~~~~~~~~~~~~~~~~~~~~

The runnable `single-teacher smoke source`_ supplies model-compatible teacher
placement, routing, generation limits and checkpoint settings. Start with it,
set ``config_groups.algorithm_recipe: opd``, and retain these objective values:

.. code-block:: yaml

   trainer:
     algorithm:
       advantage_estimator: uniform
       policy_loss_type: importance_sampling
       loss_reduction: token_mean
       use_kl_loss: false
       use_kl_in_reward: false
       use_entropy_loss: false
       advantage_batch_normalize: false
       off_policy_correction: none
       dynamic_sampling:
         type: null
       distillation:
         objective: sampled_reverse_kl
         reward_mode: replace
         coefficient: 1.0
         routing_plan: opd
         advantage_clip: null

Launch the smoke source with a pinned student model and GSM8K parquet as its file
header specifies. Its teacher revision and resource layout are already concrete.
For OPD added to RL, keep the teachers and routing, select ``grpo`` as the recipe,
and override the following fields. Explicit values in the smoke source take
precedence over the recipe, so apply all these overrides:

.. code-block:: yaml

   trainer:
     algorithm:
       advantage_estimator: grpo
       policy_loss_type: regular
       loss_reduction: sequence_mean
       use_kl_loss: false
       distillation:
         objective: sampled_reverse_kl
         reward_mode: add
         coefficient: 0.2
         routing_plan: opd
         advantage_clip: 2.0

This setup trains the group-relative policy and teacher terms with no reference
KL loss. The policy consumes :math:`A^{\rm env}+A^T`. Teacher credit is added after any
environment normalization; it does not become part of the group reward baseline.

MOPD with one teacher per domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The runnable `multi-teacher smoke source`_ defines pinned math, SWE and terminal
teachers and an exact domain-weighted data mixture. Rows carry ``teacher_route``
with one of those names. Its routing configuration is:

.. code-block:: yaml

   teacher_routing:
     opd:
       revision: snowball-mopd-ultra-smoke-v1
       routes:
         math: {teacher: math, weight: 1.0}
         swe: {teacher: swe, weight: 1.0}
         terminal: {teacher: terminal, weight: 1.0}
   data:
     sampling:
       kind: domain-weighted
       domain_weights: {math: 1.0, swe: 1.0, terminal: 1.0}

The ``mopd`` recipe follows the paper's sequence averaging (section 3.2.1) and
``advantage_clip: 5.0`` (appendix A). When adapting the smoke source, set both
``loss_reduction: sequence_mean`` and ``distillation.advantage_clip: 5.0``
explicitly when starting from that source. Its ``token_mean`` overrides the
recipe, and its standalone sampled objective leaves teacher gaps unclipped.
The existing smoke configuration keeps those choices.
For chosen-token MOPD, keep each teacher's ``evidence: chosen_token`` and use
``sampled_reverse_kl``. The clip bounds each raw teacher gap before its route
weight and coefficient; a lower route weight reduces that domain's contribution
without changing the averaging denominator.

For teacher top-K divergences, change every teacher to
``evidence: topk_distribution`` and ``top_k: 32`` and select
``sparse_forward_kl``, ``sparse_reverse_kl`` or ``sparse_jsd`` (with
``jsd_beta: 0.5`` for JSD). Use null ``advantage_clip``. For the student-selected
surrogate, change every teacher to ``evidence: student_selected_topk`` with
``top_k: 32`` and apply:

.. code-block:: yaml

   trainer:
     use_sample_packing: false
     policy:
       sequence_parallel_size: 1
       megatron_config:
         tensor_model_parallel_size: 1
         context_parallel_size: 1
     algorithm:
       loss_reduction: token_mean
       off_policy_correction: none
       distillation:
         objective: student_topk_policy_surrogate
         advantage_clip: null
   generator:
     sampling_params:
       logprobs: 32

Both variants require identical student/teacher vocabularies and vLLM local
teachers. TP/CP here refer to the student learner. Keep each teacher's supported
inference geometry from its model-specific smoke config. A sparse teacher
objective uses the teacher's support; the surrogate uses the rollout student's
support, which is why every teacher's K must equal generator ``logprobs``.

Watch ``distillation/teacher_count``, ``distillation/scored_tokens`` and
``distillation/valid_tokens`` for coverage. Chosen-token training reports
``distillation/teacher_advantage_mean``, ``distillation/teacher_advantage_abs_mean``
and ``distillation/teacher_advantage_clipped_fraction``. Top-K training reports
``distillation_loss``. Check route-specific scoring metrics, reward/evaluation,
raw gradient norm and optimizer-step diagnostics together. A finite loss alone
does not establish that every domain supplied teacher evidence.

Launch errors and remedies
--------------------------

:doc:`objective` lists the shared loss, reduction, REPLACE, geometry and
non-finite-limit checks. These additional checks apply to correction selection,
teacher evidence and group admission.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Error or requirement
     - Remedy
   * - An objective key is unsupported.
     - Use ``off_policy_correction`` and supported policy-loss settings; supply no unsupported keys.
   * - Asynchronous OLD-anchored training has null correction.
     - Select a preset, custom rules or explicit ``none``.
   * - Correction rules have no active OLD-anchored policy row.
     - Use ``none`` with ``behavior_clip``, ``sft`` or top-K REPLACE; otherwise select an OLD-anchored loss.
   * - Invalid correction name, bounds, fields or multiple truncations.
     - Use a listed preset or ``custom`` with the rule schema above; keep at most one truncate rule.
   * - Rules supplied for a preset, or missing for ``custom``.
     - Supply rules only with ``off_policy_correction: custom``.
   * - Student-top-K surrogate K mismatch or missing behavior top-K evidence.
     - Set generator ``sampling_params.logprobs`` to every teacher's ``top_k`` and preserve aligned evidence through admission.
   * - Distillation has no teachers, an unknown routing plan/teacher, or unused teacher roles.
     - Declare every referenced teacher, select the named routing plan and give each teacher a route consumer.
   * - Teacher evidence kind does not match the objective.
     - Use ``chosen_token``, ``topk_distribution`` or ``student_selected_topk`` as specified above; top-K evidence requires positive ``top_k``.
   * - Invalid distillation coefficient or optional parameters.
     - Use a positive finite coefficient; positive finite ``advantage_clip`` only for sampled reverse KL and ``entry_clip`` only for sparse forward KL; ``0 < jsd_beta < 1`` only for sparse JSD.
   * - Unsupported teacher runtime or tokenizer mismatch.
     - Apply :doc:`opd`'s backend, placement, endpoint and exact vocabulary requirements.
   * - Invalid group minimum or incomplete exact-physical group.
     - Match ``n_samples_per_prompt`` to the estimator contract; for RLOO-N choose minimum 2 through the physical group size.
   * - Invalid dynamic sampling thresholds or missing unshaped rewards.
     - Use finite nonnegative ``min_reward_std``, finite/null ``max_mean_reward`` with ``type: filter``, and provide raw outcomes for ``informative_on: unshaped``.

.. _policy loss source: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/losses.py
.. _reduction source: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/reduction.py
.. _correction source: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/correction.py
.. _preset source: https://github.com/marin-community/MarinSkyRL/tree/main/skyrl-train/skyrl_train/config/off_policy_correction
.. _advantage source: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/utils/advantage_estimators.py
.. _group filter source: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/dynamic_sampling.py
.. _teacher loss source: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/teacher.py
.. _teacher configuration source: https://github.com/marin-community/MarinSkyRL/blob/main/marinskyrl/distillation.py
.. _recipe source: https://github.com/marin-community/MarinSkyRL/tree/main/skyrl-train/skyrl_train/config/algorithm_recipe
.. _single-teacher smoke source: https://github.com/marin-community/MarinSkyRL/blob/main/cloud/iris/configs/snowball_opd_math_smoke.yaml
.. _multi-teacher smoke source: https://github.com/marin-community/MarinSkyRL/blob/main/cloud/iris/configs/snowball_mopd_ultra_smoke.yaml
.. _DeepSeekMath: https://arxiv.org/abs/2402.03300
.. _DAPO: https://arxiv.org/abs/2503.14476
.. _Understanding R1-Zero-Like Training: https://arxiv.org/abs/2503.20783
.. _Back to Basics: https://arxiv.org/abs/2402.14740
.. _REINFORCE++: https://arxiv.org/abs/2501.03262

.. _launch document schema: https://github.com/marin-community/MarinSkyRL/blob/main/cloud/iris/launch_config.py
