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
          max_removed_chars: 4096
          on_failure: error        # error | mask
          tis_reference: rollout   # rollout | none
          kl_reference: training   # training | rollout

``on_failure`` decides what happens to a sample whose first user message carries the start
marker but whose span cannot be isolated (the marker repeats, no end marker follows, or the
span is longer than ``max_removed_chars``): ``error`` raises; ``mask`` keeps its reward in the
group baseline and zeroes its loss mask. An edit failure is a property of the task text, so
under ``mask`` every rollout of that task fails alike and group admission drops the whole
group, invisibly to the per-step counters. Keep ``error`` unless a mixed pipeline needs the
run to survive a bad task. A message without the start marker is an unprompted rollout and
trains as is, so a batch can mix prompted and unprompted tasks.

``max_removed_chars`` bounds the span because the end marker is a template anchor, not the
block's end: anything the harness appends to the instruction after the block (terminus-2 adds
MCP server and skills sections there when configured) would otherwise be removed with it.

Two references follow the edit:

* ``tis_reference``. The rollout engine sampled under the prompted context, so the
  truncated-importance-sampling ratio that corrects engine-vs-trainer mismatch must compare
  logprobs computed under that same context. ``rollout`` runs one extra no-grad policy
  forward per step on the prompted sequences and re-bases the behaviour logprobs so the
  ratio ``exp(old - behaviour)`` keeps its meaning; ``none`` gives edited samples a ratio of
  exactly one and skips the forward.
* ``kl_reference``. ``training`` (default) keeps the plain self-reference. ``rollout``
  computes the frozen reference logprobs under the prompted context, so the KL term pulls
  the bare policy toward the prompted reference (the teacher); it requires
  ``kl_estimator_type: k2``, because the clamped ``k3`` estimator has zero gradient on the
  tokens where the teacher disagrees most, and it cannot be combined with
  ``use_kl_in_reward``. ``reward/policy_ref_kl`` then reports the gap to the teacher.

The feature is validated for ``policy_loss_type`` ``regular`` and ``dual_clip`` on the FSDP
backends; other losses read the behaviour logprobs directly and are rejected. With
``use_tis`` on and ``fully_async.max_staleness_steps`` above zero, ``tis_reference: none``
is rejected too, since it would drop the staleness correction on every edited row.

Metrics
~~~~~~~

``generate/context_distillation/{edited,failed,absent,removed_tokens}`` count samples and
removed prompt tokens per step. ``context_distillation/shift_per_token`` is the mean over the
edited samples' response tokens of ``log pi(y | prompted) - log pi(y | bare)``, the
per-token forward KL estimate on the sampled tokens. It is what the block still buys; when
distillation works it falls toward zero over training. ``shift_per_token_head`` and
``shift_per_token_tail`` split it at the first 512 response positions (the block acts mostly
on the first assistant turn), ``shift_per_trajectory`` is the total in nats per trajectory,
and ``edited_fraction`` the share of edited rows among the real (unpadded) rows. The shift
keys are NaN, never zero, on a step without the second forward or without an edited row.

On the first step the TIS diagnostics must look like a run without any block: with the
re-basing in place ``tis/log_ratio_abs_mean`` stays at the engine-vs-trainer level and is
unrelated to ``shift_abs_per_token``; equality between the two is the signature of a leaked
context shift.

Caveats
~~~~~~~

* The response budget of a trajectory is still set by the prompted prompt length, so an
  edited sample trains with a prompt shorter by the block and the same response.
* Evaluation never strips: a probe with the block in the task text measures the prompted
  policy, a probe without it measures what was distilled. Pair probes accordingly.
* Retained trajectory archives record the served prompt (``rollout_prompt_token_ids``);
  training-input dumps carry the bare one.
* A step whose groups were buffered before the feature was enabled trains as is (warning
  logged); a group that carries stripped prompts is re-based even if the driver's flag has
  since been turned off, and neutralised (ratio one) when the second forward is unavailable.
* terminus-2 summarization re-injects the original instruction, block included, into the
  masked context of a summarized rollout; ``generate/trajectories_summarized`` sizes it.
* With ``tis_reference: none`` the ``policy/rollout_train_prob_diff_*`` diagnostic reads one
  on edited rows by construction.
