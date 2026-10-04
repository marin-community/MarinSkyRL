Canonical task rollouts
=======================

Marin defines ``ShellboxRolloutEngine`` in
``lib/rolloutengine/src/rolloutengine/engine.py``.
SkyRL uses that engine through ``TaskRolloutWorker`` in
``skyrl_train/rollouts/task_worker.py``.

``ShellboxRolloutEngine.run`` asynchronously executes one task. The worker
starts one coroutine for each task, and inference runs on the worker's event
loop. Blocking Gym environment operations use a separate executor.
``trajectory_runner.max_concurrent_tasks`` limits active task coroutines.
If that setting is absent, the concurrency limit uses
``trajectory_runner.rollout_workers.executor_threads``.

Task and environment operations
-------------------------------

A task Parquet file contains one serialized ``TaskSpec`` per row in the
``task_spec`` column. A task declares its public conversation, executable
environment, and private grading inputs.

The default SkyRL entrypoint converts Gym source rows to this format through
``GymTaskDataset``. It writes reusable files in ``data.task_cache_dir``.
``skyrl_train.entrypoints.taskcompendium`` reads task Parquet directly.
The SWE examples use this entrypoint. The Harbor entrypoint converts task
directories and packed sources through ``HarborTaskDataset`` and uses the same worker.
With ``data.terminal_bench_data``, the default entrypoint prepares mixed Nemotron
rows through ``NemotronTaskDataset``. Terminal rows contain the complete executable
task. The worker does not require the original task directories.
Harbor settings apply only to Harbor tasks. Gym tasks retain their own turn limits
and error policies. Whole-trajectory and per-step output preserve task order,
teacher routes, and source labels.

``GymTaskSession`` creates the environment, calls ``init`` and ``step``, grades
the result, and closes its resources. It does not call the model.
The canonical engine owns the inference loop for single-turn and multi-turn tasks.
It also owns conversation and token accumulation. Each session returns its
initial messages and model options in a typed ``SessionStart`` record.
New task sessions implement environment operations without another rollout loop.

Exact tokens
------------

The model adapter uses the backend chat template and returns exact prompt and
response tokens. Each continuation retains all previously served tokens.
For example, prompt tokens ``[1, 2]`` and response tokens ``[3, 4]`` require the
next prompt to start with ``[1, 2, 3, 4]``.

The engine gives model tokens a loss mask of ``1``. It gives observation and
chat-boundary tokens a mask of ``0`` and a log probability of ``0``.
It does not reconstruct sampled responses from text or add sampled EOS tokens.
An environment cannot replace the sampled action with different text.
Token-contract violations abort the prompt group.

``generator.max_turns`` limits environment transitions.
``generator.engine_init_kwargs.max_model_len`` sets the model context limit when
configured. Each response fits the space after the exact rendered prompt.
Without that setting, ``generator.max_input_length`` limits each prompt.
A context-limit stop retains completed turns. An overlong initial prompt has no
response or grade and does not enter loss or baseline calculations.

Training data and failures
--------------------------

``WholeTaskProjection`` emits one row per rollout. ``StepTaskProjection`` emits
one row per retained model turn. Select step projection with
``trainer.step_wise_training=true``. The two projections preserve exact tokens,
behavior log probabilities, token rewards, expert routes, and teacher routes.

Verifier scores remain separate from optimization rewards. A missing or failed
verifier excludes the rollout from loss and baseline calculations. Explicitly
skipped grading retains trainable tokens with zero reward.
GenRM comparison grading completes before the worker emits a rollout group.

``generator.error_handling`` controls mask, zero-reward, and pass-through
policies. Timeout recovery retains only completed, verified Gym turns.
It requires behavior log probabilities when the request requires them.
``preserve_logprobs_on_timeout=false`` disables timeout recovery.

``TaskRolloutWorker.run_task`` projects and finalizes a completed prompt group before one
buffer write. A failed group cannot commit partial results.
``environment.skyrl_gym.max_env_workers`` limits environment threads per worker.
Cancellation waits for active environment operations before resource cleanup.
The worker returns after the buffer commit.

Rollout telemetry records collection, backend tokenization, batch assembly,
finalization, model waits, and environment queue and execution times.

Image-backed task machines
--------------------------

TaskCompendium uses ``trajectory_runner.machine.backend`` to select ``docker``
(the default) or ``qemu`` for Docker environments. This setting does not change
Harbor backend selection or task network limits.

For a verified prepared image bundle, set ``machine.runtime_bundle`` and leave
``machine.qemu.assets`` as ``null``::

    trajectory_runner:
      machine:
        backend: qemu
        runtime_bundle:
          manifest_uri: s3://<regional-bucket>/<runtime>/manifest.json
          manifest_sha256: <manifest-sha256>
          archive_uri: s3://<regional-bucket>/<runtime>/bundle.tar.gz
          archive_sha256: <archive-sha256>
          installation_parent: /opt
        qemu:
          acceleration: tcg
          assets: null

All image-backed tasks in one run must use the manifest's exact pinned ``source_image``.

Each node checks the manifest and archive hashes and extracts the prepared bundle
before Ray starts. Rollout workers check the local bundle before machine creation.
The manifest supplies ``directory_name`` and the exact pinned ``source_image``;
the factory maps that registry image to the installed bundle. Shellbox checks
its image metadata before use. This path needs no host compiler, OCI staging
tools, or host package installation. Runtime manifests must have no host packages.

For on-worker image preparation, supply ``trajectory_runner.skopeo`` and
``trajectory_runner.image_cache``, plus explicit assets::

    trajectory_runner:
      machine:
        backend: qemu
        qemu:
          acceleration: tcg
          bundle_cache: /tmp/task-bundles
          assets:
            qemu: /opt/task-runtime/qemu-system-x86_64
            kernel: /opt/task-runtime/vmlinuz
            busybox: /opt/task-runtime/busybox
            firmware: /opt/task-runtime/firmware
            libraries: /opt/task-runtime/lib
            umoci: /opt/task-runtime/umoci
            disk_size_mb: 2048
            runtime_id: <immutable-runtime-identity>

Provision these assets on each rollout worker before use. Image preparation
requires ``cc``, ``cpio``, ``mkfs.ext4`` and access to the registry. The task guest
has no network. QEMU rejects tasks that request network access, GPUs, or a
storage-size override. ``tcg`` does not need ``/dev/kvm``; ``auto`` can use KVM
when available.

Run the explicit machine acceptance check before training::

    PYTHONPATH=skyrl-train uv run --frozen python skyrl-train/scripts/check_task_machine.py \
      <config.yaml> <registry-image@sha256:digest>
