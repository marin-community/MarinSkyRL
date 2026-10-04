Canonical task rollouts
=======================

Marin defines ``ShellboxRolloutEngine`` in
``lib/rolloutengine/src/rolloutengine/engine.py``.
SkyRL uses that engine through ``TaskRolloutWorker`` in
``skyrl_train/rollouts/task_worker.py``.

``ShellboxRolloutEngine.run`` asynchronously executes one task. The worker
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

The default SkyRL entrypoint converts source rows to this format through
``SourceTaskDataset``. It writes reusable files in ``data.task_cache_dir``.
``skyrl_train.entrypoints.taskcompendium`` reads task Parquet directly.
The SWE examples use this entrypoint. The Harbor entrypoint converts task
directories and packed sources through ``HarborTaskDataset`` and uses the same worker.
With ``data.terminal_bench_data``, the default entrypoint prepares mixed Nemotron
rows through ``NemotronTaskDataset``. Terminal rows contain the complete executable
task. The worker does not require the original task directories.
Harbor settings apply only to Harbor tasks. Other tasks retain their own turn limits
and error policies. Whole-trajectory and per-step output preserve task order,
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
policies. Timeout recovery retains only completed, verified task turns.
It requires behavior log probabilities when the request requires them.
``preserve_logprobs_on_timeout=false`` disables timeout recovery.

``TaskRolloutWorker.run_task`` projects and finalizes a completed prompt group before one
buffer write. A failed group cannot commit partial results.
``environment.task_sessions.max_verifier_workers`` limits verifier threads per worker.
Cancellation waits for active verifier threads before resource cleanup.
The worker returns after the buffer commit.

Rollout telemetry records collection, backend tokenization, batch assembly,
finalization, and model waits.
