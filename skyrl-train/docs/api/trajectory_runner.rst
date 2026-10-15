Trajectory runner API
=====================

The task worker returns canonical rollouts and projects them into training batches.

Core APIs
---------

.. autoclass:: skyrl_train.rollouts.task_worker.TaskRolloutWorker
   :members:
   :member-order: bysource

See :doc:`data` for request and training batch types.

.. autoclass:: skyrl_train.inference_engines.base.InferenceEngineInterface
   :members:
   :member-order: bysource
   :undoc-members:

.. autoclass:: skyrl_train.inference_engines.inference_engine_client.InferenceEngineClient
   :members:
   :member-order: bysource
   :undoc-members:
