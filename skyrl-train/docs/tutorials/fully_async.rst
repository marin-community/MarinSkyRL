Asynchronous Training with the Rollout Buffer
=============================================

SkyRL trains every run, synchronous or asynchronous, with one training loop that works with any
``TrajectoryRunner``, including single-turn environments and multi-turn agent harnesses. Rollout workers
generate prompt groups under leases from a rollout buffer. Each training step trains on one batch of
``trainer.train_batch_size`` prompt groups, syncs the new weights to the inference engines, and publishes the new
policy step to the buffer.

A single setting, ``trainer.rollout_buffer.max_staleness_steps``, decides how far generation may run ahead of
training. At ``0`` training is synchronous and on-policy. A positive value lets generation for later batches
continue while the trainer trains, so the inference engines stay busy between steps. This is the in-flight weight
update approach of AReal and PipelineRL. ``trainer.rollout_buffer.batch_policy`` decides whether a batch still
waits for its slowest rollout.

A group is the smallest unit of data in the loop: the ``generator.n_samples_per_prompt`` trajectories generated
for one prompt.

.. contents:: Table of Contents
   :local:
   :depth: 2
   :backlinks: none

Configuration
-------------

- ``trainer.rollout_buffer.max_staleness_steps`` (default ``0``): How many policy steps may separate the step at
  which a group was leased from the step that trains on it. ``0`` is synchronous on-policy training, ``1`` is
  one-step off-policy pipelining, and larger values trade on-policy behavior for throughput. Colocated training
  (``trainer.placement.colocate_all=true``) shares GPUs between training and generation and requires ``0``.
- ``trainer.rollout_buffer.batch_policy`` (default ``full_batch``): How committed groups form batches when
  ``max_staleness_steps`` is positive. ``full_batch`` trains each step on exactly the groups leased for it;
  ``rolling`` fills each batch in commit order. See `Batch policy`_.
- ``trainer.rollout_buffer.max_in_flight`` (default ``null``): Maximum number of prompt groups generating at
  once. ``null`` bounds generation only by staleness. A value below ``trainer.train_batch_size`` generates each
  batch in several waves.
- ``trainer.rollout_buffer.object_store_root`` (default ``null``): Directory, usually an S3 prefix, that holds
  each trainable group as its own object. ``null`` keeps groups only in Ray's object store. See `Checkpointing`_.
- ``trainer.algorithm.dynamic_sampling.type``: ``filter`` discards groups without enough reward spread as they
  arrive; ``null`` keeps every group. See `Dynamic sampling`_.
- ``trainer.algorithm.group_admission.stall_timeout``: Seconds a training step may wait without admitting a group
  before training fails. The ``null`` default allows 30 minutes before any step timing exists, then adapts to
  ``max(5 * recent median step time, 10 minutes)``. Set a positive value only when the workload needs a fixed
  deadline.
- ``generator.weight_sync_pause``: Local vLLM's weight-sync pause policy. The default is ``mode: keep`` with
  ``clear_cache: true``. See `Weight sync`_.

A positive staleness needs separate GPUs for training and generation. The following snippet dedicates 4 GPUs to
each:

.. code-block:: bash

    trainer.placement.colocate_all=false \
    trainer.placement.colocate_policy_ref=true \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    generator.num_inference_engines=4 \
    generator.inference_engine_tensor_parallel_size=1 \
    trainer.rollout_buffer.max_staleness_steps=4

``examples/fully_async/async_run_gsm8k.sh`` trains Qwen2.5-1.5B-Instruct on GSM8K this way.

How It Works
------------

Leases and the staleness bound
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The trainer's dispatcher asks the buffer for a lease, takes the next prompt from the prompt loader, and hands one
task to a rollout worker. The worker generates the group and commits it to the buffer, which releases the lease.
The loader walks the dataset in a seeded order and offers prompts awaiting regeneration before new ones.

A lease records the policy step at which it was granted. The buffer grants no lease before the trainer publishes
its first policy step. After that it grants a lease only when both hold:

- fewer than ``max_in_flight`` groups are generating, and
- the groups not yet trained (leased, committed, or in the current batch) stay within
  ``max_staleness_steps + 1`` batches.

A group leased at step ``t`` must be trained by step ``t + max_staleness_steps``, so leasing more would only
produce stale work. At ``max_staleness_steps=0`` the buffer leases groups only for the current batch and, once
that batch is full, grants nothing more until the trainer publishes the next step: synchronous training.

Every lease also carries a batch id: the training step expected to train the group.

Batch policy
~~~~~~~~~~~~

If each batch takes the first groups to commit, groups that generate quickly (short answers, easy prompts, few
agent turns) train sooner and at lower staleness than slow ones. A slow group can also outlast the staleness window
and be discarded; when its prompt is generated again, the group that finally trains is one that happened to
finish quickly. The batch policy chooses between that bias and waiting for stragglers. The two policies are the
same at ``max_staleness_steps=0``.

``full_batch`` (the default)
    Leases fill the earliest batch within the staleness window that has room, and each batch trains on exactly
    the groups leased for it. Generation for the next ``max_staleness_steps`` batches runs while a batch
    completes, but groups for later batches never join it, so a step waits for its slowest group. A group is
    never discarded as stale, and neither whether nor when a group trains depends on how long it took to
    generate. A group that the content checks or dynamic sampling reject is replaced by a new lease for the same
    batch.

``rolling``
    Each committed group joins the earliest batch with room, in commit order, so a slow group never holds up
    a step. A group whose batch would be more than ``max_staleness_steps`` steps past its lease is stale: its
    prompt returns to the loader and is generated again. Prompts whose rollouts outlast the window never train.

Admission
~~~~~~~~~

The buffer assigns every committed group to a batch as it arrives, by the batch policy, and judges it against
that batch immediately:

1. Under ``rolling``, a group whose batch would be more than ``max_staleness_steps`` steps past its lease is
   stale. Its prompt returns to the loader and is generated again. Staleness is measured from the step at which
   the group was leased, the oldest policy that could have contributed to it, even when later weights produced
   some of its tokens.
2. A group that violates the run's content checks, or repeats a prompt UID already in its batch, is dropped.
3. Dynamic sampling decides the rest (see below).
4. The remaining groups join their batch. Groups admitted to a later batch reach the trainer once it publishes
   that step.

Every step trains on exactly ``trainer.train_batch_size`` groups.

Dynamic sampling
~~~~~~~~~~~~~~~~

With ``trainer.algorithm.dynamic_sampling.type=filter``, the buffer applies the filter as each group arrives and
discards groups whose rewards do not spread enough to carry a learning signal. The freed capacity leases a new
prompt, so the batch keeps filling. ``trainer.algorithm.dynamic_sampling.max_sample_batches`` limits each batch to
``max_sample_batches * train_batch_size`` candidate groups; a batch that reaches the limit without filling fails
the run. ``-1`` removes the limit.

Teacher scoring
~~~~~~~~~~~~~~~

In distillation runs, admitted groups stream to teacher scoring as they are admitted, before the batch is
complete. ``trainer.teacher_scoring.max_queued_per_teacher`` and ``trainer.teacher_scoring.workers_per_teacher``
bound each teacher's queued and concurrent requests, and a full queue holds back further admission. The trainer
assembles the training batch once every admitted group has its teacher evidence.

Weight sync
~~~~~~~~~~~

After each training step the trainer syncs the new weights to the inference engines and publishes the new policy
step. While generation runs ahead of training, the sync pauses generation first. Requests that arrive during the
pause wait for it to end. ``generator.weight_sync_pause.mode`` controls requests already running in local vLLM
engines:

- ``abort`` ends them. The client continues a non-streaming chat completion or single-prompt
  ``generate`` call from its generated tokens, so the response can contain tokens sampled under both policies.
  It re-issues a single-prompt ``/completions`` request once from the start. A streaming chat completion ends
  early with finish reason ``abort`` because a stream cannot be re-issued mid-response. Agents that stream, such
  as OpenCode under Harbor, see that turn cut short.
- ``wait`` lets them finish before syncing weights, which may hold up the training step. It requires a vLLM
  EngineCore process (``generator.vllm_v1_disable_multiproc=false``).
- ``keep`` (the default) freezes them in place and resumes them after syncing weights. Streams and batched
  requests remain active across the sync; the runner does not need to re-issue them.

A batched ``generate`` or ``/completions`` request that starts during any pause fails. A trajectory interacting
with its environment when the sync happens is unaffected.

``generator.weight_sync_pause.clear_cache=true`` (the default) clears the KV and prefix caches. Under ``keep``,
vLLM re-prefills each running request's prompt and generated tokens under the new weights. Setting it to
``false`` keeps KV from the old weights, avoiding that work but potentially mixing policies in later generation.
Only ``keep`` allows ``false``. A non-default pause policy requires local vLLM engines; SGLang and remote engines
cannot pause generation.

Checkpointing
~~~~~~~~~~~~~

A checkpoint saves the rollout state alongside the model:

- the loader position in the current pass over the dataset,
- committed groups that no trained batch has consumed, with their trajectories, or with their object URIs when
  ``object_store_root`` is set, and
- the prompts of uncommitted rollouts, together with prompts already awaiting regeneration.

On resume, the committed groups return to the buffer and the saved prompts are generated again before the loader
draws new ones. Partially generated trajectories are not saved.

References
----------

- AReal: https://arxiv.org/abs/2505.24298v3
- PipelineRL: https://arxiv.org/abs/2509.19128v2
