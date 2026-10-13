Native tasks with RolloutEngine
===============================

The standard training entrypoint executes prepared TaskSpecs through
Marin's ``ShellboxRolloutEngine`` and SkyRL's common task worker.

This entrypoint supports single-stage tasks and whole-rollout or step-wise training. Multiple
model turns can occur within one task. Shell commands use the ``docker`` backend.
Tasks without commands or shell verifiers do not require Docker machines.

Input rows
----------

Use JSON, JSONL, Parquet, or a Hugging Face dataset. Each row contains a
``task_spec`` field with a serialized ``TaskSpec``. ``TaskSpecDataset`` adds
machine selections and session settings from the run configuration. Private
verifier inputs remain inside the task record. The engine sends only public
instructions and tool observations to the model.

This example writes one machine-free arithmetic task:

.. code-block:: python

   import json
   from pathlib import Path

   from taskcompendium.grader import verifyit_package
   from taskcompendium.models import (
       AnswerType, ConversationInput, EnvironmentRequirements, PlainText, Source, TaskSpec, TextMessage,
   )
   from verifyit.spec import NumericSpec

   task = TaskSpec(
       id="arithmetic",
       context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
       environment_requirements=EnvironmentRequirements(),
       answer_type=AnswerType.NUMBER,
       answer_format=PlainText(),
       grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
       source=Source(dataset="example", revision="1", row="0", importer_revision="1"),
   )
   row = {"task_spec": task.model_dump_json()}
   Path("tasks.jsonl").write_text(json.dumps(row) + "\n")

Select the module in a training command:

.. code-block:: bash

   uv run --extra vllm --extra megatron \
     python -m skyrl_train.entrypoints.main_base \
     data.train_data='[tasks.jsonl]'

The command requires policy, inference, and topology settings. See
:doc:`../getting-started/quickstart` for a full training command.
An Iris launch recipe selects ``entrypoint: standard``.
Its ``context_budget.request_window_tokens`` sets the context window.
``context_budget.max_new_tokens_per_turn`` limits each response.
``environment.task_sessions.session`` supplies turn limits and execution deadlines.

Iris launches support machine-free tasks. Docker-backed tasks require Docker,
Skopeo, and access to a Docker daemon on the rollout worker host.
When an executable is unavailable, the worker rejects a batch whose task or verifier machine selects Docker before model requests.
Machine-free tasks do not require those executables.

Machines and grading
--------------------

``environment.task_sessions.runtime`` supplies task and verifier machine settings.
Tasks that execute shell commands select ``task_machine.backend="docker"``.
The task declares a prebuilt, digest-pinned Docker image. A shell verifier also
selects a separate ``verifier_machine`` and declares its own digest-pinned image.
See the `SWE example <../../examples/mini_swe_agent/README.md>`_ for image,
artifact, and verifier configuration.
The engine transfers declared artifacts to a fresh verifier machine after the
model turn loop. It does not install private grader files on the task machine.

``command_timeout`` applies to each command. A larger ``tool_turn_timeout`` can
limit all tool calls from one response. The engine returns command-timeout
observations to the model. Cleanup occurs outside the attempt deadline and uses
the session's finite cleanup limit.

Training output
---------------

The task worker projects ``RolloutData`` directly into training batches. It uses
SkyRL's model client, worker pool,
reward shaping, retained trajectory records, and leased rollout buffer. Exact served
tokens and behavior logprobs remain aligned. Tool observation tokens have zero
loss masks. Reconstructed model tokens cause a transport-contract error.

Correct and wrong answers train with their verifier grades. Invalid task records
fail before execution.

``generator.error_handling`` controls terminal failures. The default configuration
enables classification with ``enable_error_classification: true``. Classified
model context overflow and model timeouts receive zero optimization reward and
remain in the group baseline. Infrastructure failures are masked from loss and the baseline.
An overlong initial prompt that prevents generation has no response or verdict.
That row is excluded from loss and the baseline.
Explicit exception overrides take precedence: ``mask_exceptions``,
``zero_exceptions``, or ``passthrough_exceptions`` select the corresponding policy.
For example, ``passthrough_exceptions: [AgentTimeoutError]`` selects pass-through
for model and tool-turn timeouts. Attempt and verifier timeouts use
``TrialTimeoutError`` and ``VerifierTimeoutError``.

Setting ``preserve_logprobs_on_timeout: false`` excludes completed tokens from loss
after a timeout. The error policy still controls optimization reward and baseline
membership. Exact tokens and logprobs remain in the evidence.

Pass-through requires an available verifier score and any required behavior
logprobs. Otherwise, the row is masked from loss and the baseline. Verifier scores
remain separate from optimization rewards. Without an execution failure,
unavailable verdicts are masked from loss and the baseline. With an execution
failure, the exception policy applies. Explicitly skipped grading can use a session's reward.

An empty response becomes one fully masked token in the trainer row. Its behavior
logprob is zero. The original rollout evidence remains unchanged.

Prepare source rows and Harbor packages as TaskSpecs before training.
The trainer does not extract task archives or join source datasets.

For source rows that already contain prompts and private grader inputs, use an
explicit preparation command:

.. code-block:: bash

   uv run python -m marinskyrl.task_preparation \
     --input raw.parquet --output tasks.parquet --source my-dataset \
     --config grader.json --environment task-environment.json

``grader.json`` contains task grader parameters. ``task-environment.json``
contains ``EnvironmentRequirements``. Omit these options for tasks that use
neither extra grader parameters nor a machine environment.
Select the session factory in the run config. The preparation command does not
select a backend, session factory, or deadline.
The trainer does not convert source rows or open task directories.
All generated RL uses the same worker, with retries and group grading before projection.
Static DPO uses a separate chosen/rejected dataset path.
Multi-stage tasks and task-specific image builds are not supported.
See :doc:`task_rollouts` for session and backend selection.
