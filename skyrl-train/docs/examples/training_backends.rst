Training with Megatron
======================

Megatron is the training backend. The default configuration sets ``trainer.strategy=megatron``.
Install the frozen Megatron and vLLM extras before launching a GPU run:

.. code-block:: bash

    uv sync --frozen --extra megatron --extra vllm --group dev
    cd skyrl-train
    NUM_GPUS=4 bash examples/gsm8k/run_gsm8k.sh

Set tensor, pipeline, context, and expert parallel sizes under
``trainer.policy.megatron_config`` and ``trainer.ref.megatron_config``. See
:ref:`megatron-configurations` and :doc:`megatron` for model-specific recipes.
