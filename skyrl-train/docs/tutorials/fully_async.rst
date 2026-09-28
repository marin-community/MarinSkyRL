Asynchronous Training with the Rollout Buffer
=============================================

SkyRL trains every run, synchronous or asynchronous, with one training loop that works with any
``TrajectoryRunner``, including single-turn environments and multi-turn agent harnesses. Rollout workers
generate prompt groups under leases from a rollout buffer. Each training step trains on one batch of
``trainer.train_batch_size`` prompt groups, syncs the new weights to the inference engines, and publishes the new
policy step to the buffer.

A single setting, ``trainer.rollout_buffer.max_staleness_steps``, decides how far generation may run ahead of
training. At ``0`` training is synchronous and on-policy. A positive value lets generation continue while the
trainer trains, which removes the stalls that long or straggling rollouts cause in synchronous training. This is
the in-flight weight update, or partial rollout, approach of AReal and PipelineRL.

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
- ``trainer.rollout_buffer.max_in_flight`` (default ``null``): Maximum number of prompt groups generating at
  once. ``null`` bounds generation only by staleness. A value below ``trainer.train_batch_size`` generates each
  batch in several waves.
- ``trainer.algorithm.dynamic_sampling.type``: ``filter`` discards groups without enough reward spread as they
  arrive; ``null`` keeps every group. See `Dynamic sampling`_.
- ``trainer.algorithm.group_admission.stall_timeout``: Seconds a training step may wait without admitting a group
  before training fails. The ``null`` default allows 30 minutes before any step timing exists, then adapts to
  ``max(5 * recent median step time, 10 minutes)``. Set a positive value only when the workload needs a fixed
  deadline.

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

The trainer's dispatcher asks the buffer for a lease, takes the next prompt from the group loader, and hands one
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

Admission
~~~~~~~~~

The buffer judges every committed group immediately, in arrival order, against the current step's batch:

1. A group whose lease is more than ``max_staleness_steps`` steps older than the current policy step is stale.
   Its prompt returns to the loader and is generated again. Staleness is measured from the step at which the
   group was leased, the oldest policy that could have contributed to it, even when later weights produced some
   of its tokens. A slow group can become stale when groups leased after it fill the batches it could have joined.
2. A group that violates the run's content checks, or repeats a prompt UID already in the batch, is dropped.
3. Dynamic sampling decides the rest (see below).
4. The remaining groups join the batch. Groups that arrive after the batch is full wait and are judged again
   against the next step.

Every step trains on exactly ``trainer.train_batch_size`` groups.

Dynamic sampling
~~~~~~~~~~~~~~~~

With ``trainer.algorithm.dynamic_sampling.type=filter``, the buffer applies the filter as each group arrives and
discards groups whose rewards do not spread enough to carry a learning signal. The freed capacity leases a new
prompt, so the batch keeps filling. ``trainer.algorithm.dynamic_sampling.max_sample_batches`` limits each step to
``max_sample_batches * train_batch_size`` candidate groups; a step that reaches the limit without a full batch
fails. ``-1`` removes the limit.

Teacher scoring
~~~~~~~~~~~~~~~

In distillation runs, admitted groups stream to teacher scoring as they are admitted, before the batch is
complete. ``trainer.teacher_scoring.max_queued_per_teacher`` and ``trainer.teacher_scoring.workers_per_teacher``
bound each teacher's queued and concurrent requests, and a full queue holds back further admission. The trainer
assembles the training batch once every admitted group has its teacher evidence.

Weight sync
~~~~~~~~~~~

After each training step the trainer syncs the new weights to the inference engines and publishes the new policy
step. While generation runs ahead of training, the sync pauses generation first. Pausing holds every engine's
scheduler: in-flight requests stop where they are and later requests wait. On resume, in-flight requests continue
from the tokens they already generated, now under the new weights. The runner never observes the pause, so any
trajectory runner works without changes, whether a trajectory is mid-generation or interacting with its
environment.

Checkpointing
~~~~~~~~~~~~~

A checkpoint saves the rollout state alongside the model:

- the loader position in the current pass over the dataset,
- committed groups that no trained batch has consumed, with their trajectories, and
- the prompts of uncommitted rollouts, together with prompts already awaiting regeneration.

On resume, the committed groups return to the buffer and the saved prompts are generated again before the loader
draws new ones. Partially generated trajectories are not saved.

References
----------

- AReal: https://arxiv.org/abs/2505.24298v3
- PipelineRL: https://arxiv.org/abs/2509.19128v2
