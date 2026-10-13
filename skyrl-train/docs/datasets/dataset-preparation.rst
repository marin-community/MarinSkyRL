Prepared task datasets
======================

Generated RL accepts bare ``TaskSpec`` records. Prepare raw source rows before the
training command. The trainer does not infer task families or convert Harbor packages.
Static DPO uses its separate chosen/rejected format.

Input format
------------

Each row contains ``task_spec``: a JSON string that satisfies the TaskCompendium
``TaskSpec`` schema. Optional ``teacher_route`` and ``metric_tags`` columns contain
training labels, not task requirements. A teacher route selects a configured teacher
for distillation. Metric tags supply coverage labels.
Curriculum sampling uses ``extra_info.data_source`` and ``extra_info.grade`` for bin labels.
These fields contain no verifier answers.
An optional ``data_source`` column supplies the training source label when a row
uses a task from another source. Runtime source overrides still match ``TaskSpec.source.dataset``.

.. code-block:: python

   row = {"task_spec": task.model_dump_json(), "teacher_route": "math", "metric_tags": ["math"]}

The loader accepts prepared Parquet, JSON, JSONL, or Hugging Face datasets. It derives
public prompts from the TaskSpec for length filtering. It does not write an intermediate
dataset. Task IDs must be unique within a split.

Source preparation
------------------

``marinskyrl.task_preparation`` converts source chat rows to bare TaskSpecs.
It keeps answers and grader parameters in private verifier resources.
It does not select the session factory, machine backend, or execution limits.
Different session types require distinct source names.
The preparation CLI rejects different ``env_class`` values under one source name.
Harbor preparation rejects duplicate task IDs before it writes execution overrides.

.. code-block:: bash

   uv run --project .. python -m marinskyrl.task_preparation \
     --input raw.parquet --output tasks.parquet --source my_dataset \
     --config grader.json

``grader.json`` contains grader parameters, not credentials. Use ``--environment``
with an ``EnvironmentRequirements`` JSON file when the task requires a machine.
Native executables require Linux process semantics and a prebuilt, digest-pinned image.
The source builder must also declare the workspace resources that the task requires.

The GSM8K, multiplication, multi-turn math, SearchR1, NUPA, and code examples write
prepared TaskSpecs directly. Code preparation requires an explicit Python image.
Use ``marinskyrl.harbor_preparation`` for supported Harbor packages and
``marinskyrl.nemotron_preparation`` for mixed Nemotron rows.
These commands run before training, not in a rollout worker.

Run settings
------------

Select the session factory and runtime settings in the run config:

.. code-block:: bash

   uv run --project .. -m skyrl_train.entrypoints.main_base \
     data.train_data="['tasks.parquet']" \
     data.val_data="['validation_tasks.parquet']" \
     environment.task_sessions.session.task_session=gsm8k

``environment.task_sessions.sources`` supplies overrides keyed by
``TaskSpec.source.dataset``. Each override contains ``runtime`` and ``session``
settings. SkyRL preserves the TaskSpec when it creates a ``LoweredTaskSpec``.
Unknown source keys cause rejection before GPU allocation.
The driver matches these keys against the loaded train and evaluation task sources.
Native graders use the Shellbox session. Custom graders use the selected session factory.
See :doc:`../tutorials/task_rollouts` for machine settings and deadlines.
