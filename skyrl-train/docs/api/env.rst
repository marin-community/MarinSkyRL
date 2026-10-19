Task rollout API
================

``StepHarness`` drives episodes with ``prepare``, ``advance``, and ``close`` operations.
``Transition.data`` carries SkyRL's per-turn reward and grade. The engine owns model
inference, token records, and the final task grade.

.. autoclass:: rolloutengine.harness.step.StepHarness
   :members:
   :member-order: bysource

.. autoclass:: rolloutengine.models.Prompt
   :members:

.. autoclass:: rolloutengine.models.Transition
   :members:

.. autoclass:: rolloutengine.models.ModelTurn
   :members:

.. autoclass:: taskcompendium.models.TaskSpec

.. autoclass:: taskcompendium.models.EnvironmentRequirements

.. autoclass:: rolloutengine.models.MachineRuntimeSpec

.. autoclass:: rolloutengine.models.RolloutLimits

.. autoclass:: rolloutengine.models.TaskRolloutSpec

.. autoclass:: skyrl_gym.task_records.TaskTransition

.. autoclass:: skyrl_gym.source_task.SessionParameters
