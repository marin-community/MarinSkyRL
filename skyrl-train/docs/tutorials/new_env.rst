Create a task session
=====================

A task session holds task state and defines the response to each model turn.
The rollout engine calls the model and records the conversation and exact tokens.
Shellbox supplies an execution machine when the task requires one.

The multiplication example requires no machine. It reads a private reference answer
and returns correction prompts until the model gives the correct answer or reaches the turn limit.

Session interface
-----------------

Implement the four operations in Marin's ``TaskSession`` protocol:

- ``prepare()`` returns the initial messages and model options in ``SessionStart``.
- ``advance(turn)`` returns a ``Transition`` with observations, a reward, and terminal state.
- ``grade(messages)`` returns the final ``GradeResult``.
- ``close()`` releases session resources. The engine then closes the supplied machine.

A ``ModelTurn`` contains the sampled assistant message, exact token IDs, and log probabilities.
Use its message or text for task actions. Do not replace the sampled tokens.
A session can return ``reset_conversation`` when the next attempt requires a fresh conversation.

The multiplication implementation is:

.. literalinclude:: ../../examples/multiply/task_session.py
   :language: python
   :pyobject: MultiplyTaskSession

Intermediate turns receive zero reward and a correction prompt.
A terminal correct answer in ``\boxed{42}`` format receives 1.0.
A terminal wrong boxed answer receives 0.5.
An answer with no box receives zero.
The final grade is the mean of all completed turn grades, including intermediate zeros.
For example, turn grades of 0.0 and 1.0 give a final grade of 0.5.
The training projection retains the individual turn rewards.

Supply the factory
------------------

Pass the session class to the experiment:

.. code-block:: python

   from examples.multiply.task_session import MultiplyTaskSession
   from skyrl_train.entrypoints.main_base import BasePPOExp

   experiment = BasePPOExp(cfg, sessions={"multiply": MultiplyTaskSession})
   experiment.run()

A factory accepts ``(TaskSpec, Machine | None)`` and returns a fresh session.
The worker sends this factory map to its Ray workers.

Prepare source rows
-------------------

Each source row declares its task name and private reference:

.. code-block:: python

   row = {
       "prompt": [{"role": "user", "content": "6 * 7"}],
       "env_class": "multiply",
       "reward_spec": {"method": "rule", "ground_truth": "42"},
       "data_source": "synthetic_multiply",
       "max_turns": 5,
   }

``SourceTaskDataset`` converts these rows directly to serialized tasks in the prepared dataset.
The prompt contains public messages. The verifier payload contains the private reference and task configuration.
Keep the reference out of model-visible observations.

``TaskSpec.context`` holds the public conversation.
``TaskSpec.environment`` contains an ``EnvironmentSpec`` that selects the machine and named session factory.
The session decodes ``TaskSpec.verifier.parameters_json`` with ``ExternalVerifierSpec.model_validate_json``.
Use the resulting verifier's ``parameters`` mapping:
``config`` contains task settings and ``extras`` contains the source row's private fields.
The multiplication session reads ``extras["reward_spec"]["ground_truth"]``.
See :doc:`../api/env` for these types and :doc:`task_rollouts` for the Parquet format.

From ``skyrl-train/``, prepare the example data and run the supplied entrypoint:

.. code-block:: bash

   uv run --project .. examples/multiply/multiply_dataset.py --output_dir "$HOME/data/multiply"
   bash examples/multiply/run_multiply.sh

Execution and rewards
---------------------

For an executable task, declare ``machine`` in
``environment.task_sessions.<task_name>``.
Use ``Machine.run``, ``upload``, and ``download`` for task execution.
The engine owns the machine lifecycle. See :doc:`tools_guide`.

A transition reward contributes to the final token of that model turn.
Observation tokens have zero loss mask and zero log probability.
A missing or failed grade excludes the rollout from training.
Explicitly skipped grading retains trainable model tokens.

The source label also supplies ``reward/domain/<source>/avg_raw_reward`` metrics.
Missing labels use ``_missing``. Each batch records at most 32 source keys.
Additional sources contribute to ``reward/domain_overflow/avg_raw_reward``.

See :doc:`task_rollouts` for exact-token and failure policies.
