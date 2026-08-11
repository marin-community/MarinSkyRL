Algorithm Registry API
=====================================

The registry system in SkyRL Train provides a way to register and manage custom algorithm functions (like advantage estimators and policy loss functions) across distributed Ray environments. This system allows users to extend the framework with custom implementations without modifying the core codebase.

Base Registry Classes
---------------------

.. autoclass:: skyrl_train.utils.function_registry.BaseFunctionRegistry
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.utils.function_registry.RegistryActor
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autofunction:: skyrl_train.utils.algorithm_registry.sync_registries
    
Advantage Estimator Registry
-----------------------------

The advantage estimator registry manages functions that compute advantages and returns for reinforcement learning algorithms.

.. autoclass:: skyrl_train.utils.algorithm_registry.AdvantageEstimatorRegistry
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.utils.algorithm_registry.AdvantageEstimator
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autofunction:: skyrl_train.utils.algorithm_registry.register_advantage_estimator


Policy Loss Registry
--------------------

The policy loss registry manages functions that compute policy losses for PPO and related algorithms.

.. autoclass:: skyrl_train.utils.algorithm_registry.PolicyLossRegistry
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autoclass:: skyrl_train.utils.algorithm_registry.PolicyLossType
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. autofunction:: skyrl_train.utils.algorithm_registry.register_policy_loss
