Megatron Backend for 5D Parallelism
===================================

SkyRL supports NVIDIA's `Megatron-Core <https://developer.nvidia.com/megatron-core>`_ library as an RL training backend, inheriting support for 5D parallelism (tensor+sequence, pipeline, context, expert, and data parallelism), and optimized performance for large scale models.

We provide example scripts for running efficient large scale MoE training with models like ``Qwen3-30B-A3B`` using Megatron in the `examples/megatron <https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples/megatron>`_ directory.
For details on configuring the Megatron backend, and enabling checkpointing, see :ref:`megatron-configurations`, and :ref:`megatron-checkpointing`.


Parallelism with Megatron
-------------------------

Megatron combines data, tensor, pipeline, context, and expert parallelism. Configure these
sizes under ``trainer.policy.megatron_config`` and ``trainer.ref.megatron_config``.
The examples cover Qwen3-30B-A3B and Qwen3-235B-A22B.

A script for running the Qwen3-30B-A3B experiment can be found `here <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/megatron/run_search_megatron.sh>`_. 
Additionally, we provide a script for running basic GSM8K training on Qwen3-235B-A22B with Megatron `here <https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/megatron/run_megatron_qwen3-235b-a22b.sh>`_. 
Note that although training at 100B+ scale using Megatron is currently possible, we are in the process of further optimizing peformance.

For more details on configuring the Megatron backend, and enabling checkpointing, see :ref:`megatron-configurations`, and :ref:`megatron-checkpointing`.

.. _megatron-installation:

Installation
------------

The root project lock contains the Megatron and vLLM runtime closure. From the repository root:

.. code-block:: bash

    uv sync --frozen --extra megatron --extra vllm --group dev

Configuration
-------------
We provide the following options for fully configuring the Megatron backend, exposing the underlying Megatron optimizer, DDP, and model config objects
for advanced users to fully take advantage of all of Megatron-Core's feature flags. For more details, see the :ref:`megatron-configurations` section.

.. code-block:: yaml
    :caption: ``skyrl_train/config/megatron/policy.yaml``

    # @package megatron_config.policy
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    expert_model_parallel_size: 1
    expert_tensor_parallel_size: null

    ddp_config: # pass-through config to Megatron's `DistributedDataParallelConfig` object
      # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/distributed/distributed_data_parallel_config.py#L8
      ...
    optimizer_config_kwargs: # pass-through kwargs to Megatron's `OptimizerConfig` object
      # any overlapping arguments with those we attempt to resolve in trainer.policy.optimizer_config will be overridden by the values here
      # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/optimizer/optimizer_config.py#L12
      ...
    transformer_config_kwargs: # pass-through kwargs to the Megatron's `TransformerConfig` object
      # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.13.0/megatron/core/transformer/transformer_config.py#L33
      ...
    # flag to manually empty torch's cuda cache between the forward/backward pass and the optimizer step
    # this will free reserved but unallocated memory, and can help avoid OoMs in the optimizer
    empty_cuda_cache: true

These default values can be overridden by passing in the corresponding arguments to ``trainer.policy.megatron_config`` in the launch script.

.. _parallelism-resources:

Parallelism Resources
----------------------
Understanding and configuring parallelism strategies for large models can be challenging.
Some helpful resources for understanding and tuning large scale parallelism strategies can be found at the `Huggingface Ultra-Scale Playbook <https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=finding_the_best_training_configuration>`_, 
the `The Mesh Parallelism Zoo <https://blog.ezyang.com/2025/08/the-parallelism-mesh-zoo/>`_, and the `Visualizing 6-D Parallelism <https://main-horse.github.io/posts/visualizing-6d>`_.

Below, we show a diagram displaying how all 5 parallelism strategies - tensor, pipeline, context, expert, and data parallelism - can be utilized in SkyRL, as well as how dispatching data across these parallel groups works.

.. image:: images/parallelism.svg


