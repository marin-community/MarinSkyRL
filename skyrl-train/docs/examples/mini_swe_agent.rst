SWE tasks with Shellbox
======================

SWE-Gym and SWE-Bench use ``skyrl_train.entrypoints.taskcompendium`` and the
common Shellbox rollout engine. The model calls the shell tool to change the
repository. The grader transfers the resulting Git patch to a fresh task
machine and runs the private evaluation script.

Prepare the dataset
-------------------

The source datasets are ``SumanthRH/SWE-Gym-Subset`` and
``SumanthRH/SWE-bench_Verified``. Use the package installation commands in
:code_link:`examples/mini_swe_agent/README.md`, then run from ``skyrl-train``:

.. code-block:: bash

    uv run --no-sync --project .. examples/mini_swe_agent/preprocess_swegym.py \
      --train_revision TRAIN_DATASET_COMMIT \
      --eval_revision EVAL_DATASET_COMMIT \
      --output_dir ~/data/swe_gym_subset

Replace the revision values with pinned dataset commit IDs. The output Parquet
files contain one serialized ``TaskSpec`` per row. The task includes the image,
working directory, private evaluation script, and source revision.

Run training
------------

Prepare a Docker daemon and the task images on each rollout worker. Install
TaskCompendium and Shellbox in the worker environment. Then run an example:

.. code-block:: bash

    bash examples/mini_swe_agent/run_mini_swe_8B.sh

Use ``run_mini_swe_30B.sh`` for the two-node example. The scripts select the
common entrypoint. They do not start Mini-SWE-Agent or a model HTTP proxy.
See :code_link:`examples/mini_swe_agent/README.md` for environment setup, grading
behavior, and configuration.
