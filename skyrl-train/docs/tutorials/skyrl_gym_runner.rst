Canonical task rollouts
=======================

Marin defines ``RolloutEngine`` and ``ShellboxRolloutEngine`` in
``lib/taskcompendium/src/taskcompendium/rollout.py``.
SkyRL uses that engine through ``TaskRolloutWorker`` in
``skyrl_train/rollouts/task_worker.py``.

``RolloutEngine.generate`` returns a synchronous iterator. The worker executes
each iterator in a dedicated thread pool. Inference requests run on the worker's
async event loop. Inference I/O uses a separate executor so blocked rollout
threads cannot prevent model requests from completing.
``trajectory_runner.max_concurrent_tasks`` limits the engine thread pool.
If that setting is absent, the pool uses
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
Repeated cancellation requests also wait for the engine thread to release its
task resources. The worker returns after the buffer commit.

Rollout telemetry records collection, backend tokenization, batch assembly,
finalization, model waits, and environment queue and execution times.
