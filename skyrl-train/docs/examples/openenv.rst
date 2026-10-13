OpenEnv task sessions
=====================

OpenEnv uses the shared RolloutEngine and a Shellbox machine per task.
The session starts an HTTP server inside the machine.
Reset and step results become model observations and turn rewards.
A server terminal result or the session turn limit ends the rollout.
Server failures remain infrastructure failures.

The session factory is explicit:

.. code-block:: python

   from skyrl_gym.openenv_tasks import OpenEnvTaskSession
   from skyrl_train.entrypoints.main_base import BasePPOExp

   experiment = BasePPOExp(cfg, sessions={"openenv": OpenEnvTaskSession})

Prepare tasks
-------------

The image must contain the selected server and its dependencies.
Save the environment requirements in ``environment.json``:

.. code-block:: json

   {
     "command_semantics": "linux_process",
     "docker_image": "IMAGE@sha256:DIGEST",
     "working_directory": "/app"
   }

Replace the image reference with a full, digest-pinned reference.
Save the private server parameters in ``server.json``:

.. code-block:: json

   {
     "server_command": ["python", "-m", "uvicorn", "envs.echo_env.server.app:app", "--host", "127.0.0.1", "--port", "8000"],
     "server_port": 8000
   }

From ``skyrl-train``, prepare the Echo tasks:

.. code-block:: bash

   uv run --project .. integrations/openenv/prepare_dummy_dataset.py \
     --output_dir "$HOME/data/openenv/echo_env" --env_name echo_env \
     --environment /path/to/environment.json --config /path/to/server.json

Each output row contains a bare TaskSpec.
Public context contains the action instruction.
Private verifier data contains the task type and server command.
Prepare each task type with its own image and server parameters.
The example builder supplies Echo and Coding tasks.
Other task types require benchmark-specific TaskSpec preparation.

Run tasks
---------

The example entrypoint registers ``OpenEnvTaskSession``.
Select the session and machine settings in the run config:

.. code-block:: bash

   bash integrations/openenv/run_openenv.sh \
     environment.task_sessions.runtime.task_machine.backend=docker \
     environment.task_sessions.runtime.task_machine.cpus=1 \
     environment.task_sessions.runtime.task_machine.memory_mb=1024

Set ``generator.max_turns`` for the turn limit.
The TaskSpec declares the image. The run config declares execution settings.
The host does not require the OpenEnv SDK or exposed container ports.

Supported action forms are text for Echo, Python code for Coding, integer actions
for OpenSpiel, Atari, and SUMO, and numeric lists for FinRL.
Each model response puts its action inside ``<action>...</action>``.
The server accepts ``POST /reset``, ``POST /step``, and ``GET /health``.
The final grade is the mean of completed turn rewards.
See :doc:`../tutorials/task_rollouts` for projection and cleanup.
