Choosing an RL objective
========================

Choose the advantage estimator, policy loss, averaging and rollout correction
independently. Recipes supply compatible settings; explicit launch YAML values
override them. Teacher objectives supply policy advantages or a separate term
in the loss. :doc:`objective` describes the architecture and numerical contracts;
:doc:`opd` covers teacher deployment.

Policy losses
-------------

Set these keys under ``trainer.algorithm``.
`skyrl_train/objective/losses.py <https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/losses.py>`_
defines all ten losses.

``regular``
~~~~~~~~~~~

Use token-level PPO clipping for reward-based policy updates.

.. code-block:: yaml

   policy_loss_type: regular
   eps_clip_low: 0.2
   eps_clip_high: 0.2

``dual_clip``
~~~~~~~~~~~~~

Use PPO clipping with an extra bound on negative-advantage penalties.

.. code-block:: yaml

   policy_loss_type: dual_clip
   eps_clip_low: 0.2
   eps_clip_high: 0.2
   clip_ratio_c: 3.0

``importance_sampling``
~~~~~~~~~~~~~~~~~~~~~~~

Use an unclipped current/old-policy ratio for advantage-weighted updates; rollout correction is independent.

.. code-block:: yaml

   policy_loss_type: importance_sampling

``behavior_clip``
~~~~~~~~~~~~~~~~~

Clip against the rollout policy for stale data; requires rollout logprobs and no separate correction.

.. code-block:: yaml

   policy_loss_type: behavior_clip
   eps_clip_low: 0.2
   eps_clip_high: 0.2
   clip_ratio_c: 3.0
   off_policy_correction: none

``gspo``
~~~~~~~~

Use sequence-ratio clipping for sequence-level credit; requires sequence averaging and no token-varying credit. `GSPO`_.

.. code-block:: yaml

   policy_loss_type: gspo
   loss_reduction: sequence_mean
   eps_clip_low: 0.0003
   eps_clip_high: 0.0004

``cispo``
~~~~~~~~~

Use detached clipped importance weights with token log-likelihood gradients. `MiniMax-M1`_.

.. code-block:: yaml

   policy_loss_type: cispo
   cispo:
     cispo_eps_clip_low: 1.0
     cispo_eps_clip_high: 5.0

``sapo``
~~~~~~~~

Use a smooth sigmoid gate for advantage-weighted updates.

.. code-block:: yaml

   policy_loss_type: sapo
   sapo:
     tau_pos: 1.0
     tau_neg: 1.05

``clip_cov``
~~~~~~~~~~~~

Suppress selected high-covariance tokens; selection depends on the microbatch.

.. code-block:: yaml

   policy_loss_type: clip_cov
   clip_cov:
     clip_ratio: 0.0002
     clip_cov_lb: 1.0
     clip_cov_ub: 5.0

``kl_cov``
~~~~~~~~~~

Penalize selected high-covariance tokens; requires an environment policy term and depends on the microbatch.

.. code-block:: yaml

   policy_loss_type: kl_cov
   kl_cov:
     kl_cov_frac: 0.2
     ppo_kl_coef: 1.0

``sft``
~~~~~~~

Maximize likelihood on eligible response tokens; advantages and teacher credit do not affect this loss.

.. code-block:: yaml

   policy_loss_type: sft
   advantage_estimator: uniform
   off_policy_correction: none

Averaging modes
---------------

Set ``trainer.algorithm.loss_reduction``; formulas and examples are in
:doc:`objective`, implemented by `objective/reduction.py`_.

.. list-table::
   :header-rows: 1

   * - Setting
     - Use
   * - ``token_mean``
     - Give each eligible data-weighted token equal weight.
   * - ``sequence_mean``
     - Give each nonempty response equal weight.
   * - ``seq_mean_token_sum_norm``
     - Normalize response sums by the configured total sequence length for Dr.GRPO.
   * - ``seq_mean_token_sum_norm_global``
     - Count only responses with nonzero policy advantage; incompatible with top-K teacher terms.

Rollout corrections
-------------------

Set ``trainer.algorithm.off_policy_correction`` for active old-policy-anchored
policy terms. These detached weights affect only the policy numerator.
Implementation: `objective/correction.py`_; configurations: `correction presets`_.

.. list-table::
   :header-rows: 1

   * - Setting
     - Use
   * - ``tis``
     - Weight tokens by old-policy/rollout-policy probability ratios capped at 2.
   * - ``icepop``
     - Retain ratio weights inside [0.5, 5] and assign zero outside.
   * - ``seq_mask_tis``
     - Keep sequences with geometric-mean ratios in [0.99, 1.01], with token ratio weights capped at 2.
   * - ``outlier_mask``
     - Give unit weight only to responses whose every eligible token ratio is in [1e-4, 100].
   * - ``none``
     - Explicitly select uncorrected updates, including asynchronous baselines.
   * - ``null``
     - Leave correction unspecified; invalid for asynchronous old-policy-anchored training.
   * - ``custom``
     - Supply token/sequence masks and at most one ratio-bearing truncate rule.

Custom rule fields are checked with OmegaConf structured schemas. Bounds must
be positive and finite, with low <= high. Truncation requires high and no low;
masks require a bound. Sequence aggregates are ``geometric``, ``product`` or
``extreme_token``; the last supports masks only.

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

Advantage estimators and filtering
----------------------------------

Set ``trainer.algorithm.advantage_estimator``. Implementation:
`utils/advantage_estimators.py`_.

.. list-table::
   :header-rows: 1

   * - Setting
     - Use
   * - ``uniform``
     - Supply unit credit for likelihood training or teacher REPLACE objectives.
   * - ``reward``
     - Use response reward sums without a prompt-group baseline.
   * - ``grpo``
     - Use a prompt-relative baseline with optional ``grpo_norm_by_std: false``. `DeepSeekMath`_.
   * - ``rloo``
     - Use a leave-one-out baseline over a complete prompt group. `Back to Basics`_.
   * - ``rloo_n``
     - Exclude infrastructure failures; set ``group_advantage_min_size`` between 2 and the physical group size.
   * - ``rloo_n_pbs``
     - Add potential-based token shaping to an eligible-response baseline; requires an explicit group minimum of at least 2.
   * - ``reinforce++``
     - Use critic-free discounted returns; ``gamma: 1.0`` gives undiscounted returns. `REINFORCE++`_.
   * - ``gae``
     - Use temporal credit with a critic; configure ``trainer.critic.model.path``, ``gamma`` and ``lambd``.

Use dynamic sampling to retain informative final trial groups; the mean-reward
ceiling is exclusive. Implementation: `dynamic_sampling.py`_.

.. code-block:: yaml

   dynamic_sampling:
     type: filter
     informative_on: unshaped
     min_reward_std: 0.0
     max_mean_reward: 0.9

Teacher objectives
------------------

Set ``trainer.algorithm.distillation.objective`` and supply teachers, routing,
``coefficient`` and ``reward_mode: add`` or ``replace``. Implementation:
`objective/teacher.py`_; configuration requirements: :doc:`opd` and :doc:`objective`.

.. list-table::
   :header-rows: 1

   * - Setting
     - Use
   * - ``sampled_reverse_kl``
     - Train from chosen-token teacher advantages; optional ``advantage_clip`` bounds raw gaps before weighting. `MOPD`_, section 3.2.1.
   * - ``sparse_forward_kl``
     - Match the teacher's conditional top-K distribution; optional ``entry_clip`` bounds individual summands.
   * - ``sparse_reverse_kl``
     - Match selected teacher tokens plus aggregate tail mass; partial-support teacher tails use a 1e-12 floor.
   * - ``sparse_jsd``
     - Match a teacher/student mixture over top-K plus tail; set ``0 < jsd_beta < 1``.
   * - ``student_topk_policy_surrogate``
     - Use a clipped surrogate on rollout-selected support; requires token averaging, exact sampled tokens and matching generator/teacher K.

Recipes
-------

Select ``config_groups.algorithm_recipe`` in a launch YAML, or
``+algorithm_recipe=NAME`` with Hydra. `Recipe configs`_ give the complete settings.

.. list-table::
   :header-rows: 1

   * - Recipe
     - Use and source
   * - ``grpo``
     - Group-relative credit, sequence averaging, clip 0.2/0.2 and KL coefficient 0.04. `DeepSeekMath`_.
   * - ``dapo``
     - Token averaging, clip 0.2/0.28, reward-spread filtering and KL off. `DAPO`_.
   * - ``dr_grpo``
     - Mean-centered group credit, fixed-length normalization and KL off. `Understanding R1-Zero-Like Training`_.
   * - ``gspo``
     - Sequence-ratio clipping at 0.0003/0.0004 with sequence averaging and explicit KL off. `GSPO`_.
   * - ``cispo``
     - Detached importance weights in [0, 6], token averaging and KL off. `MiniMax-M1`_.
   * - ``opd``
     - Chosen-token teacher REPLACE credit with token averaging and KL off. `MOPD`_, section 3.2.1.
   * - ``mopd``
     - Routed teacher REPLACE credit, sequence averaging, advantage clip 5.0 and KL off. `MOPD`_.

The GSPO paper's objective has no KL term; section 2 says it omits the term for
brevity and reports no coefficient. This recipe explicitly sets
``use_kl_loss: false`` and ``use_kl_in_reward: false`` rather than inheriting the
base default. Set reference KL and its coefficient explicitly when wanted.

End-to-end examples
-------------------

GRPO and DAPO
~~~~~~~~~~~~~

Use this ``skyrl`` subtree in a `launch document`_ with staged model inputs and
two nodes of one GPU each; prepare GSM8K parquet as in
:doc:`../datasets/dataset-preparation`.

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

Set ``algorithm_recipe: dapo`` for DAPO. Its overlong filtering and reward
shaping are generator settings outside the recipe; configure
``generator.apply_overlong_filtering`` and
``generator.trajectory_reward_shaping.overlong`` with the generation limits.
Existing GSM8K and DAPO examples have their own objective settings; explicit
values in those sources override recipe defaults.

The enclosing launch document supplies ``run``, ``runtime``, ``iris``, ``ray``,
``artifacts`` and ``inputs``. Save it as ``launch.yaml`` and submit from the
repository root:

.. code-block:: bash

   uv run --frozen python -m cloud.iris.launch iris launch --config launch.yaml

Asynchronous training with correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the GRPO setup above with separate policy/rollout placement and these overrides:

.. code-block:: yaml

   trainer:
     rollout_buffer:
       max_staleness_steps: 2
     algorithm:
       policy_loss_type: regular
       off_policy_correction: tis

The launcher requests required rollout logprobs; admission checks their alignment
and finiteness. Asynchronous old-policy-anchored training requires an explicit
correction choice, including ``none`` for an uncorrected baseline.

OPD alone and added to RL
~~~~~~~~~~~~~~~~~~~~~~~~~

Start from the runnable `single-teacher smoke config`_, which provides pinned
teacher placement and routing. Supply the student model and GSM8K parquet as its
header specifies, select ``config_groups.algorithm_recipe: opd``, and apply:

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

For teacher credit added to RL, retain the teachers and routing, select the
``grpo`` recipe, and apply these explicit overrides:

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

MOPD with one teacher per domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start from the runnable `multi-teacher smoke config`_ with its pinned math, SWE
and terminal teachers, route-labeled parquet and domain-weighted sampler. Select
``config_groups.algorithm_recipe: mopd`` and set these overrides explicitly:

.. code-block:: yaml

   trainer:
     algorithm:
       loss_reduction: sequence_mean
       distillation:
         objective: sampled_reverse_kl
         reward_mode: replace
         coefficient: 1.0
         routing_plan: opd
         advantage_clip: 5.0
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

Each response needs its ``teacher_route``; every teacher uses
``evidence: chosen_token`` and shares the student's token-ID vocabulary.
Sequence averaging and advantage clip 5.0 follow MOPD section 3.2.1 and appendix A.
The smoke source's explicit objective settings take precedence over recipes,
so both overrides above are required.

For sparse divergences, use ``evidence: topk_distribution`` and ``top_k: 32``
on every teacher, select the objective and set ``advantage_clip: null``;
JSD also needs ``jsd_beta: 0.5``. For the student-selected surrogate, use
``evidence: student_selected_topk`` with ``top_k: 32`` and apply:

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

All top-K objectives require student TP=CP=sequence parallelism=1, sample
packing disabled, local vLLM teachers and identical teacher/student vocabularies.
Keep each teacher's supported inference geometry. The student-selected surrogate
also requires exact sampled completion tokens and generator ``logprobs`` equal
to every teacher's K. The teacher-only REPLACE setup preserves configured
sampling, including nucleus probability 0.99 in native Open-MOPD. An active
rollout-anchored or corrected policy term requires temperature-only sampling.
Deployment requirements are in :doc:`opd`.

.. _objective/reduction.py: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/reduction.py
.. _objective/correction.py: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/correction.py
.. _correction presets: https://github.com/marin-community/MarinSkyRL/tree/main/skyrl-train/skyrl_train/config/off_policy_correction
.. _utils/advantage_estimators.py: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/utils/advantage_estimators.py
.. _dynamic_sampling.py: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/dynamic_sampling.py
.. _objective/teacher.py: https://github.com/marin-community/MarinSkyRL/blob/main/skyrl-train/skyrl_train/objective/teacher.py
.. _Recipe configs: https://github.com/marin-community/MarinSkyRL/tree/main/skyrl-train/skyrl_train/config/algorithm_recipe
.. _launch document: https://github.com/marin-community/MarinSkyRL/blob/main/cloud/iris/launch_config.py
.. _single-teacher smoke config: https://github.com/marin-community/MarinSkyRL/blob/main/cloud/iris/configs/snowball_opd_math_smoke.yaml
.. _multi-teacher smoke config: https://github.com/marin-community/MarinSkyRL/blob/main/cloud/iris/configs/snowball_mopd_ultra_smoke.yaml
.. _DeepSeekMath: https://arxiv.org/abs/2402.03300
.. _DAPO: https://arxiv.org/abs/2503.14476
.. _Understanding R1-Zero-Like Training: https://arxiv.org/abs/2503.20783
.. _Back to Basics: https://arxiv.org/abs/2402.14740
.. _REINFORCE++: https://arxiv.org/abs/2501.03262
.. _GSPO: https://arxiv.org/abs/2507.18071
.. _MiniMax-M1: https://arxiv.org/abs/2506.13585
.. _MOPD: https://arxiv.org/abs/2606.30406
