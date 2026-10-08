Native tasks with RolloutEngine
===============================

The opt-in ``rollout_engine`` entrypoint executes native TaskCompendium tasks
through Marin's ``ShellboxRolloutEngine``. The existing Gym, Harbor, and mini-SWE
entrypoints do not change.

This entrypoint supports single-stage tasks and whole-rollout training. Multiple
model turns can occur within one task. Shell commands use the ``docker`` backend.
Tasks without commands or shell verifiers do not require Docker machines.

Input rows
----------

Use the existing ``PromptDataset`` input formats: JSON, JSONL, Parquet, or a
Hugging Face dataset. Each row contains:

* ``prompt``: the public conversation from ``TaskSpec.context``.
* ``lowered_task_json``: serialized ``LoweredTaskSpec``.
* Optional ``env_class``: a metrics label, not a Gym environment selection.

The prompt must equal ``conversation_messages(task.context)``. Private verifier
inputs remain inside the task record. The engine sends only public instructions
and tool observations to the model.

This example writes one machine-free arithmetic task:

.. code-block:: python

   import json
   from pathlib import Path

   from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec, TaskSessionSpec
   from taskcompendium.grader import grader_package
   from taskcompendium.models import (
       AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage,
   )
   from taskcompendium.submission import conversation_messages
   from verifyit.spec import NumericSpec

   task = TaskSpec(
       id="arithmetic",
       context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
       environment_requirements=EnvironmentRequirements(),
       answer_type=AnswerType.NUMBER,
       verifier=grader_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).verifier,
       source=Source(dataset="example", revision="1", row="0", importer_revision="1"),
   )
   lowered = LoweredTaskSpec(
       task=task,
       runtime=TaskRuntimeSpec(task_machine=None, verifier_machine=None),
       session=TaskSessionSpec(
           task_session="shellbox",
           max_turns=1,
           model_turn_timeout=60,
           command_timeout=None,
           tool_turn_timeout=None,
           total_turn_timeout=60,
           attempt_timeout=90,
           verifier_timeout=30,
           cleanup_timeout=10,
       ),
   )
   row = {
       "prompt": conversation_messages(task.context),
       "env_class": "arithmetic",
       "lowered_task_json": lowered.model_dump_json(),
   }
   Path("tasks.jsonl").write_text(json.dumps(row) + "\n")

Select the module in an existing whole-rollout training command:

.. code-block:: bash

   uv run --extra vllm --extra megatron \
     python -m skyrl_train.entrypoints.rollout_engine \
     data.train_data='[tasks.jsonl]'

The command requires the same policy, inference, and topology settings as the
other training entrypoints. An Iris launch recipe selects ``entrypoint:
rollout_engine``. The recipe's context budget supplies the model context window
and per-response token limit. The task record supplies session turn limits and
execution deadlines.

Iris launches support machine-free tasks. Docker-backed tasks require access to
a Docker daemon on the rollout worker host.

Machines and grading
--------------------

Records that execute shell commands select ``task_machine.backend="docker"``.
The task declares a prebuilt, digest-pinned Docker image. A shell verifier also
selects a separate ``verifier_machine`` and declares its own digest-pinned image.
The engine transfers declared artifacts to a fresh verifier machine after the
model turn loop. It does not install private grader files on the task machine.

``command_timeout`` applies to each command. A larger ``tool_turn_timeout`` can
limit all tool calls from one response. The engine returns command-timeout
observations to the model. Cleanup occurs outside the attempt deadline and uses
the record's finite cleanup limit.

Training output
---------------

The adapter uses SkyRL's existing model client, worker pool, whole-rollout
projection, reward shaping, retention, and leased rollout buffer. Exact served
tokens and behavior logprobs remain aligned. Tool observation tokens have zero
loss masks. Reconstructed model tokens cause a transport-contract error.

Correct and wrong answers train with their verifier grades. Invalid task records
fail before execution.

``generator.error_handling`` controls terminal failures. The default configuration
enables classification with ``enable_error_classification: true``. Model context
overflow and model timeouts receive zero optimization reward and remain in the
group baseline. Infrastructure failures are masked from loss and the baseline.
Explicit exception overrides take precedence: ``mask_exceptions``,
``zero_exceptions``, or ``passthrough_exceptions`` select the corresponding policy.

Pass-through requires an available verifier score and any required behavior
logprobs. Otherwise, the row is masked from loss and the baseline. Verifier scores
remain separate from optimization rewards. Without a terminal failure, skipped
or unavailable verdicts are masked from loss and the baseline.

An empty response becomes one fully masked token in the trainer row. Its behavior
logprob is zero. The original rollout evidence remains unchanged.

This entrypoint does not convert existing Gym source rows or Harbor roots.
It does not support custom session factories, step-wise training, multi-stage
tasks, task-specific image builds, retries, or group-level grading. Use the
existing entrypoints for those workflows.
