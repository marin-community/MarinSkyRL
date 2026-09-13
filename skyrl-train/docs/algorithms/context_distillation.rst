Context Distillation
====================

Context distillation trains a policy to reproduce behaviour it only shows under an extra
instruction, without that instruction. It is the training half of
`P²O <https://arxiv.org/abs/2603.21877>`_: rollouts are sampled under a guidance block
appended to the task, and the policy gradient is computed on the bare task. P²O's ablation
shows that computing the gradient on the prompted context instead makes the policy depend on
the prompt and score below plain GRPO at evaluation.

How the rollout carries the block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data side appends the block to the task instruction behind a fixed delimiter. With the
terminus-2 agent the first user message then reads::

    ... Task Description:
    <instruction>

    ---
    Working guidance:
    <block>

    Current terminal state:
    ...

The trainer never sees the block text. It knows the two anchors and removes everything from
the start marker up to the end marker from the first user message before it tokenizes the
training prompt. Blocks can differ per task or per rollout; nothing in the trainer changes.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    trainer:
      algorithm:
        context_distillation:
          enabled: true
          start_marker: "\n\n---\nWorking guidance:\n"
          end_marker: "\n\nCurrent terminal state:"
          on_failure: mask        # mask | error
          tis_reference: rollout  # rollout | none
          kl_reference: rollout   # rollout | training

``on_failure`` decides what happens to a sample whose first user message carries the start
marker but whose span cannot be isolated (the marker repeats, or no end marker follows):
``mask`` keeps its reward in the group baseline and zeroes its loss mask; ``error`` raises.
A message without the start marker is an unprompted rollout and trains as is, so a batch
can mix prompted and unprompted tasks.

Two references follow the edit:

* ``tis_reference``. The rollout engine sampled under the prompted context, so the
  truncated-importance-sampling ratio that corrects engine-vs-trainer mismatch must compare
  logprobs computed under that same context. ``rollout`` runs one extra no-grad policy
  forward per step on the prompted sequences and re-bases the behaviour logprobs so the
  ratio ``exp(old - behaviour)`` keeps its meaning; ``none`` gives edited samples a ratio of
  exactly one and skips the forward.
* ``kl_reference``. ``rollout`` computes the frozen reference logprobs under the prompted
  context, so the KL term pulls the bare policy toward the prompted reference (the teacher).
  ``training`` keeps the plain self-reference.

Metrics
~~~~~~~

``generate/context_distillation/{edited,failed,absent,removed_tokens}`` count samples and
removed prompt tokens per step. ``context_distillation/shift_per_token`` is the mean over the
edited samples' response tokens of ``log pi(y | prompted) - log pi(y | bare)``, the
per-token forward KL estimate on the sampled tokens. It is what the block still buys; when
distillation works it falls toward zero over training. ``shift_per_trajectory`` is the same
in nats per trajectory and ``edited_fraction`` the share of edited rows in the batch.

Caveats
~~~~~~~

* The response budget of a trajectory is still set by the prompted prompt length, so an
  edited sample trains with a prompt shorter by the block and the same response.
* Evaluation never strips: a probe with the block in the task text measures the prompted
  policy, a probe without it measures what was distilled. Pair probes accordingly.
* Retained trajectories and data dumps decode the training prompt; the served prompt is
  kept as ``rollout_prompt_token_ids`` in the trajectory batch for audits.
