Canonical task rollouts
=======================

``TaskRolloutWorker`` runs TaskCompendium tasks through Marin's ``ShellboxRolloutEngine``.
The engine owns model calls, conversation and exact-token accumulation, deadlines, and resource cleanup.
The worker controls concurrency, retries, group grading, and training projections.

``ShellboxRolloutEngine.run(lowered)`` asynchronously executes one task. The worker
starts one coroutine for each task, and inference runs on the worker's event
loop. Synchronous graders use a separate executor. Shellbox commands use asynchronous machine operations.
``trajectory_runner.max_concurrent_tasks`` limits active task coroutines.
Its default is 32 per worker. The verifier thread count does not set task concurrency.

Task and environment operations
-------------------------------

A task Parquet file contains one serialized ``LoweredTaskSpec`` per row in the
``lowered_task_spec`` column. The record preserves a ``TaskSpec`` and adds machine selections and session limits.
The task declares public context, tools, private grading inputs, resources, and environment requirements.

``skyrl_train.entrypoints.main_base`` converts source rows through
``SourceTaskDataset`` with Hugging Face ``Dataset.map``.
It retains serialized tasks in memory without an intermediate Parquet or Arrow cache file.
Source-row conversion does not use ``data.task_cache_dir``.
``skyrl_train.entrypoints.taskcompendium`` reads task Parquet directly.
The `SWE example <../../examples/mini_swe_agent/README.md>`_ uses this entrypoint with ``data.train_data`` and ``data.val_data``.
``skyrl_train.entrypoints.terminal_bench`` converts task
directories and packed sources through ``HarborTaskDataset`` and uses the same worker.
Harbor caches private task Parquet in ``data.task_cache_dir``.
Explicit exports use ``skyrl_train.dataset.tasks.write_tasks(Path(...), records)``.
``LoweredTaskSpec.runtime`` selects optional task and verifier machines through ``MachineRuntimeSpec``.
Each machine selection supplies a configured backend identifier, network policy, hardware limits, user, and startup or cleanup deadline.
``LoweredTaskSpec.session`` supplies the session factory identifier, turn limit, model and command caps, tool-turn deadline, cumulative turn deadline, attempt deadline, and verifier deadline.
Its cleanup deadline requires an explicit finite, positive value.
Tasks support one stage and prebuilt, digest-pinned images. Multi-stage packages and task-specific image builds cause rejection.
TaskCompendium defines task serialization. SkyRL owns its dataset file format.
With ``data.terminal_bench_data``, ``skyrl_train.entrypoints.main_base`` prepares mixed Nemotron
rows through ``NemotronTaskDataset``. Terminal rows contain the executable task and its execution settings.
The worker does not require the original task directories.
``harbor.max_turns`` and Harbor exception settings apply only to Harbor tasks.
These exception settings include ``harbor.mask_exceptions`` and ``harbor.default_error_treatment``.
Other tasks use their lowered session limits and ``generator.error_handling``.
See :doc:`../datasets/dataset-preparation` for source rows and training inputs.
Whole-trajectory and per-step output preserve task order,
teacher routes, and source labels. A teacher route identifies the configured teacher model for distillation of that row.

The worker supplies explicit factories from ``skyrl_gym/task_factories.py``.
Each factory creates a direct implementation of Marin's ``TaskSession`` protocol.
The session prepares the task, executes model actions, returns observations, and grades the result.
The engine creates its Shellbox machine and closes the session before the machine.
Pure answer graders use ``runtime.task_machine=None`` and do not create a machine.
Each session returns its initial messages and model options in ``SessionStart``.

The worker registers the native ``docker`` backend only when Docker and the configured Skopeo executable are available.
It validates each ``LoweredTaskSpec`` record's backend and session before it starts a batch.
An unavailable backend aborts the batch before model requests and buffer writes.
The per-row infrastructure error policy does not mask this deployment error.
Machine-free tasks do not require these executables, even when unused session configuration selects Docker.
Executable availability does not establish Docker daemon access or image contents.

``environment.task_sessions.session`` supplies launch-time session limits.
Source-specific ``session`` blocks override those limits, including ``max_turns``.
Source conversion uses a non-null row ``max_turns``, then ``extra_info.max_turns``, then the common session limit.
An explicit source-family ``session.max_turns`` overrides the row limit.
The common session limit uses ``generator.max_turns`` by default.
Materialized task records keep their stored session limits.
For example, ``environment.task_sessions.lcb.session.total_turn_timeout`` overrides the cumulative turn deadline for code tasks.
Harbor lowering uses package machine settings, users, total-turn deadlines, and verifier deadlines.
Other Harbor session limits come from launch configuration.
The attempt deadline includes startup, preparation, all turns, and final verification.
The cumulative turn deadline excludes preparation and final verification.
The verifier deadline includes artifact transfer and separate verifier startup.
Cleanup runs outside the attempt deadline. Machine cleanup can override the session cleanup limit.
``command_timeout`` limits each shell command. A command timeout returns a ``timed_out`` tool observation, and the model can continue.
``tool_turn_timeout`` limits the full ``advance`` call. Its budget is separate from the command limit.
When these limits are finite, lowering requires the command limit to be less than the tool-turn deadline.
Shell grading requires a prebuilt, digest-pinned verifier image, a separate verifier machine, and declared artifacts for workspace submissions.
The Harbor importer accepts only separate verifier environments.
An unset verifier mode without a separate environment selects shared mode and causes rejection.
Harbor machine preparation executes commands with ``user="0"`` (root).
The Shellbox Iris backend rejects per-command user overrides, so imported Harbor tasks cannot use that backend.
Disabling Harbor verification removes private grader resources and selects skipped grading before runtime lowering.

Exact tokens
------------

The model adapter uses the backend chat template and returns exact prompt and
response tokens. Each continuation retains all previously served tokens.
For example, prompt tokens ``[1, 2]`` and response tokens ``[3, 4]`` require the
next prompt to start with ``[1, 2, 3, 4]``.

The engine gives model tokens a loss mask of ``1``. It gives observation and
chat-boundary tokens a mask of ``0`` and a log probability of ``0``.
It does not reconstruct sampled responses from text or add sampled EOS tokens.
Sessions cannot replace sampled actions with different text.
Token-contract violations abort the prompt group.

One transition follows each model response, including the final response.
The engine uses each lowered record's ``session.max_turns``.
Source conversion applies the row and source-family limits described above.
The worker applies ``harbor.max_turns`` to Harbor tasks.
A session can finish earlier.
``generator.max_input_length`` limits each rendered prompt, including continuations.
The full sequence limit is that prompt bound plus ``sampling_params.max_generate_length``.
``generator.engine_init_kwargs.max_model_len`` can reduce the full sequence limit.
Each response fits the space after the exact rendered prompt.
Sequence-normalized losses use the same full sequence bound.
A context-limit stop retains completed turns. An overlong initial prompt has no
response or grade and does not enter loss or baseline calculations.

Training data and failures
--------------------------

``WholeTaskProjection`` emits one row per rollout. ``StepTaskProjection`` emits
one row per retained model turn. Select step projection with
``trainer.step_wise_training=true``. The two projections preserve exact tokens,
behavior log probabilities, token rewards, expert routes, and teacher routes.

Sessions can supply per-turn optimization rewards. The whole-task projection sums these rewards.
Otherwise, it uses the task grade.
Math, multiplication, search, SearchCode, and SQL sessions keep the final task verdict separate from optimization rewards.
Their task grade does not average tool or correction turns. Format rewards do not count as correct answers.
Step projection uses each turn's reward, with the task grade on the last turn when no per-turn rewards exist.
A no-grade verifier result with ``RolloutData.failure=None`` excludes tokens from loss and baseline calculations.
Recorded execution failures, including verifier timeouts, use the exception policy.
They receive zero optimization reward when no grade is available.
Explicitly skipped grading can use a session's optimization reward.
Without a session reward, the row is masked from loss and the baseline.
``harbor.verifier_disable=true`` skips grading for Harbor tasks and masks those rows.
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
Retained trajectory files omit private task records, grader parameters, and references.
They also omit source metadata that contains grader inputs.
They retain public source labels, teacher routes, prompts, responses, and verifier results.

``generator.error_handling`` controls exception policies.
``mask`` excludes the rollout from loss and baseline calculations.
``zero`` retains trainable tokens with zero reward.
``passthrough`` retains the available verifier score.
Exception lists override the built-in error categories. ``default_error_treatment``
selects one of these policies for unknown errors.
Failed attempts retain exact tokens, behavior log probabilities, and available verifier grades.
Candidate Python crashes, execution timeouts, and output overflow produce zero-reward candidate failures.
Provider transport failures, Python startup failures, and model template failures use the infrastructure error policy.
Exact-token contract violations remain fatal.
The effective ``sampling_params.logprobs`` setting determines the probability requirement, including request overrides.
When that setting requests log probabilities, loss eligibility requires one log probability per retained generated token.
``generator.error_handling.preserve_logprobs_on_timeout=false`` masks loss after a timeout.
The error policy still controls optimization reward and baseline membership.
Empty responses use one fully masked placeholder token in the trainer row.

``TaskRolloutWorker.run_task`` projects and finalizes a completed prompt group before one
buffer write. A failed group cannot commit partial results.
``environment.task_sessions.max_verifier_workers`` limits verifier threads per worker.
Cancellation returns control to the engine without an unbounded wait for active verifier threads.
Session cleanup retains pending thread operations before it releases their resources.
The cleanup deadline bounds the caller's wait. Unfinished cleanup remains owned until it completes.
Worker shutdown cancels active requests, including group grading and buffer submission.
``trajectory_runner.shutdown_timeout`` bounds shutdown. Its default is 60 seconds.
Half of that budget permits request cleanup. The remaining budget permits machine cleanup.
Late machine creation uses the same owned close task as normal cleanup.
If shutdown exceeds its budget, provider diagnostics identify open creations and machines, and shutdown reports a failure.
The search task's HTTP client can use ten attempts, each with ``environment.task_sessions.search.timeout``, plus 45 seconds of retry delays.
Those attempts can continue in retained cleanup after a task deadline expires.
The worker returns after the buffer commit.

Rollout telemetry records collection, backend tokenization, batch assembly,
finalization, and model waits.
See :doc:`../api/trajectory_runner` for the worker API.
