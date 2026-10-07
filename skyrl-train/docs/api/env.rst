Task session API
================

``TaskSession`` defines task operations. ``Transition`` contains observations,
terminal state, rewards, and grading results. The engine owns model inference and token records.

.. autoclass:: rolloutengine.contracts.TaskSession
   :members:
   :member-order: bysource

.. autoclass:: rolloutengine.contracts.SessionStart
   :members:

.. autoclass:: rolloutengine.contracts.Transition
   :members:

.. autoclass:: rolloutengine.contracts.ModelTurn
   :members:

.. autoclass:: taskcompendium.models.TaskSpec

.. autoclass:: taskcompendium.models.EnvironmentRequirements

.. autoclass:: rolloutengine.spec.MachineRuntimeSpec

.. autoclass:: rolloutengine.spec.TaskRuntimeSpec

.. autoclass:: rolloutengine.spec.TaskSessionSpec

.. autoclass:: rolloutengine.spec.LoweredTaskSpec

.. autoclass:: taskcompendium.importers.skyrl.ExternalVerifierSpec
