Canonical task rollouts
=======================

Marin defines ``ShellboxRolloutEngine`` in
``lib/rolloutengine/src/rolloutengine/engine.py``.
SkyRL uses that engine through ``TaskRolloutWorker`` in
``skyrl_train/rollouts/task_worker.py``.

``ShellboxRolloutEngine.run(task, execution=...)`` asynchronously executes one task. The worker
starts one coroutine for each task, and inference runs on the worker's event
loop. Synchronous graders use a separate executor. Shellbox commands use asynchronous machine operations.
``trajectory_runner.max_concurrent_tasks`` limits active task coroutines.
If that setting is absent, the concurrency limit uses
``trajectory_runner.rollout_workers.executor_threads``.

Task and environment operations
-------------------------------

A task Parquet file contains one serialized ``TaskSpec`` per row in the
``task_spec`` column. A task declares its public conversation, executable
environment, and private grading inputs.

``skyrl_train.entrypoints.main_base`` converts source rows through
``SourceTaskDataset`` with Hugging Face ``Dataset.map``.
It retains serialized tasks in memory without an intermediate Parquet or Arrow cache file.
Source-row conversion does not use ``data.task_cache_dir``.
``skyrl_train.entrypoints.taskcompendium`` reads task Parquet directly.
The `SWE example <../../examples/mini_swe_agent/README.md>`_ uses this entrypoint with ``data.train_data`` and ``data.val_data``.
``skyrl_train.entrypoints.main_harbor`` converts task
directories and packed sources through ``HarborTaskDataset`` and uses the same worker.
Harbor caches private task Parquet in ``data.task_cache_dir``.
Explicit exports use ``skyrl_train.dataset.tasks.write_tasks(Path(...), records)``.
Each ``TaskRecord`` pairs a ``TaskSpec`` with separate ``TaskExecution`` settings.
The ``task_execution`` column stores deadlines, agent users, and stage preparation.
Task-only datasets can omit this column. Their execution settings have no time limits.
Staged tasks require an execution record with one entry for every stage.
The worker rejects a missing or unknown stage entry before inference.
TaskCompendium defines task serialization. SkyRL owns its dataset file format.
With ``data.terminal_bench_data``, ``skyrl_train.entrypoints.main_base`` prepares mixed Nemotron
rows through ``NemotronTaskDataset``. Terminal rows contain the executable task and its execution settings.
The worker does not require the original task directories.
``harbor.max_turns`` and Harbor exception settings apply only to Harbor tasks.
These exception settings include ``harbor.mask_exceptions`` and ``harbor.default_error_treatment``.
Other tasks use ``generator.max_turns`` and ``generator.error_handling``.
Whole-trajectory and per-step output preserve task order,
teacher routes, and source labels.

The worker supplies explicit factories from ``skyrl_gym/task_factories.py``.
Each factory creates a direct implementation of Marin's ``TaskSession`` protocol.
The session prepares the task, executes model actions, returns observations, and grades the result.
The engine creates its Shellbox machine and closes the session before the machine.
Pure answer graders use null environments and do not create a machine.
The canonical engine owns the inference loop for single-turn and multi-turn tasks.
It also owns conversation and token accumulation. Each session returns its
initial messages and model options in a typed ``SessionStart`` record.
New task sessions implement task operations without another rollout loop.

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

One transition follows each model response, including the final response.
The engine uses ``generator.max_turns`` unless Harbor supplies its own turn limit.
A session can finish earlier.
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

Sessions can supply per-turn optimization rewards. Otherwise, the whole-task projection uses the task grade.
Step projection uses each turn's reward, with the task grade on the last turn when no per-turn rewards exist.
A verifier result without a grade excludes tokens from loss and baseline calculations.
Recorded execution failures use the exception policy, with zero optimization reward when no grade is available. Explicitly
skipped grading retains trainable tokens with zero reward.
``harbor.verifier_disable=true`` skips grading for Harbor tasks, including all their stages.
Nemotron GenRM tasks use a judge model to compare a group of responses against
a private grading principle. Configure that judge in
``environment.task_sessions.nemotron_ultra.genrm``.
Comparison grading completes before the worker emits a rollout group.
Each training prompt group contains ``generator.n_samples_per_prompt`` attempts of one task.
GenRM compares attempts that the exception policy permits for loss calculations.
An attempt with an execution failure also requires retained trainable tokens and the requested log probabilities.
The completed rollout group enters the buffer.
Ineligible attempts receive no comparison score.
Evaluation exports contain public prompts, responses, labels, scores, and failure fields.
Private task records and grading configuration remain outside these exports.

``generator.error_handling`` controls exception policies.
``mask`` excludes the rollout from loss and baseline calculations.
``zero`` retains trainable tokens with zero reward.
``passthrough`` retains the available verifier score.
Exception lists override the built-in error categories. ``default_error_treatment``
selects one of these policies for unknown errors.
Timeout recovery retains completed turns with available grades and valid token evidence.
The effective ``sampling_params.logprobs`` setting determines the probability requirement, including request overrides.
When that setting requests log probabilities, recovery requires one log probability per retained generated token.
``generator.error_handling.preserve_logprobs_on_timeout=false`` disables timeout recovery.

``TaskRolloutWorker.run_task`` projects and finalizes a completed prompt group before one
buffer write. A failed group cannot commit partial results.
``environment.task_sessions.max_verifier_workers`` limits verifier threads per worker.
Cancellation waits for active verifier threads before resource cleanup.
Synchronous HTTP operations must finish before cancellation releases their resources.
The search task's HTTP client can use ten attempts, each with ``environment.task_sessions.search.timeout``, plus 45 seconds of retry delays.
Cancellation can wait for these attempts.
The worker returns after the buffer commit.

Rollout telemetry records collection, backend tokenization, batch assembly,
finalization, and model waits.
