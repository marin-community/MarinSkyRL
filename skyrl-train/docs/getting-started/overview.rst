SkyRL System Overview
=====================

SkyRL breaks the RL stack into modular components and provides public APIs for each of them. 

SkyRL separates training into a trainer, trajectory runners, model clients, and environments, with a controller managing setup and execution.

.. figure:: images/system-overview.png
   :alt: SkyRL System Overview
   :align: center
   :width: 80%

The components' responsibilities are as follows:

Trainer
~~~~~~~
Performs the optimization steps based on configured RL algorithm. Updates model parameters based on generated trajectories and their assigned rewards.

- `Base Training Worker interface <https://github.com/NovaSky-AI/SkyRL/blob/1c6ff519fe3b06cb8afd1ed6846348373d227bea/skyrl-train/skyrl_train/workers/worker.py#L180>`_

  - `MegatronWorker <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/skyrl_train/workers/megatron/megatron_worker.py>`_

- `PPORayActorGroup <https://github.com/NovaSky-AI/SkyRL/blob/5a82809e218b2e0c3dd431377fb672e35ecc4a84/skyrl-train/skyrl_train/workers/worker.py#L385>`_: Our abstraction for a group of training workers (as Ray actors) that jointly execute operations for a given model (e.g., policy model, critic model, etc.).

Rollout worker
~~~~~~~~~~~~~~
Marin's rolloutengine package defines the shared Shellbox rollout engine.
SkyRL supplies inference, converts its records to training batches, and writes completed groups to the rollout buffer.
Direct task sessions supply task operations and grading. Shellbox machines execute task tools.

- ``TaskRolloutWorker`` in ``skyrl_train/rollouts/task_worker.py``
- ``TaskSession`` in Marin's ``rolloutengine/contracts.py``

InferenceEngine
~~~~~~~~~~~~~~~
Executes inference on the policy model to produce model outputs (i.e., the RL agent's actions). Typically, multiple InferenceEngines are deployed to process prompts in parallel.

- `Base InferenceEngine interface <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/skyrl_train/inference_engines/base.py>`_
- `InferenceEngineClient to manage multiple engines <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/skyrl_train/inference_engines/inference_engine_client.py>`_
- `vLLM backend <https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/skyrl_train/inference_engines/vllm>`_
- `SGLang backend <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/skyrl_train/inference_engines/sglang/sglang_server.py>`_


Task session
~~~~~~~~~~~~
A task session holds task state, executes model actions, and returns observations and grades.
The engine owns model inference and the exact conversation/token record.
Shellbox supplies the execution machine when the task requires one.

- :doc:`Task session API <../api/env>`
- :doc:`Create a task session <../tutorials/new_env>`


Controller
~~~~~~~~~~
Manages physical placement, initialization, and control flow of training execution for each of the above components.

- The training control loop currently sits in `trainer.py <https://github.com/NovaSky-AI/SkyRL/blob/1c6ff519fe3b06cb8afd1ed6846348373d227bea/skyrl-train/skyrl_train/trainer.py#L128>`_
- It is a WIP to move the control loop to a separate component for even greater flexibility.
