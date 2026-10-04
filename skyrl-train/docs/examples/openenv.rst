OpenEnv task sessions
=====================

``OpenEnvTaskSession`` implements the common task interface.
Shellbox creates one machine for each rollout.
The session starts the configured OpenEnv server in that machine and sends
HTTP reset and step requests from inside the machine.

The initial server observation enters the model conversation.
Each action produces a reward and, for a nonterminal turn, an observation.
A server terminal result or the turn limit ends the rollout.
An invalid candidate action receives -1 and correction feedback.
Server failures remain infrastructure failures.

Supported action formats are:

- ``echo_env``: text inside ``<action>...</action>``.
- ``coding_env``: Python code inside those tags. Multiline code is permitted.
- ``openspiel-env`` and ``atari-env``: an integer action ID, for example ``<action>2</action>``.
- ``sumo-rl-env``: an integer phase ID inside the action tags.
- ``finrl-env``: a list of numeric actions, for example ``<action>[0.1, -0.2]</action>``.

The session factory is explicit:

.. code-block:: python

   from functools import partial
   from skyrl_gym.openenv_tasks import OpenEnvTaskSession
   from skyrl_train.entrypoints.main_base import BasePPOExp

   experiment = BasePPOExp(
       cfg,
       sessions={"openenv": partial(OpenEnvTaskSession, max_turns=cfg.generator.max_turns)},
   )

Machine configuration
---------------------

Supply the machine image and server command. For the HTTP Echo image:

.. code-block:: yaml

   environment:
     task_sessions:
       openenv:
         machine:
           kind: docker
           image:
             kind: registry
             reference: ghcr.io/meta-pytorch/openenv-echo-env:sha-64d4b10
           workdir: /app
           network: false
           memory_mb: 1024
           cpus: 1
         server_command:
           - python
           - -m
           - uvicorn
           - envs.echo_env.server.app:app
           - --host
           - 127.0.0.1
           - --port
           - "8000"
         server_port: 8000

This integration uses the HTTP API from the selected image revision.
The server must accept ``POST /reset`` with an empty object and ``POST /step``
with ``{"action": <task payload>, "timeout_s": <integer>}``.
Each response contains an ``observation`` object, a numeric or null ``reward``, and a boolean ``done``.
``GET /health`` must return a successful HTTP status before the startup deadline.
Select images and commands with this API.
The host does not require the OpenEnv Python SDK or exposed container ports.

For mixed task types, use ``machines`` keyed by the row's ``env_name``.
Each row selects the ``openenv`` factory and an image-specific application:

.. code-block:: python

   row = {
       "prompt": [{"role": "user", "content": "Return the action text."}],
       "env_class": "openenv",
       "env_name": "echo_env",
   }

.. code-block:: yaml

   environment:
     task_sessions:
       openenv:
         machines:
           echo_env:
             kind: docker
             image:
               kind: registry
               reference: ghcr.io/meta-pytorch/openenv-echo-env:sha-64d4b10
             workdir: /app
             env:
               OPENENV_APP: envs.echo_env.server.app:app
           coding_env:
             kind: docker
             image:
               kind: registry
               reference: ghcr.io/meta-pytorch/openenv-coding-env:sha-64d4b10
             workdir: /app
             env:
               OPENENV_APP: envs.coding_env.server.app:app
         server_command:
           - python
           - -c
           - 'import os, uvicorn; uvicorn.run(os.environ["OPENENV_APP"], host="127.0.0.1", port=8000)'

The importer removes ``machine`` and ``machines`` from the private session configuration.
It stores the selected machine in ``TaskSpec.environment``.
The remaining settings, including the server command, stay in the private verifier payload.

Prepare and launch
------------------

From ``skyrl-train/``:

.. code-block:: bash

   uv run --project .. integrations/openenv/prepare_dummy_dataset.py \
     --output_dir "$HOME/data/openenv/echo_env" --env_name echo_env
   bash integrations/openenv/run_openenv.sh --config-name ppo_base_config \
     +environment.task_sessions.openenv.machine.kind=docker \
     +environment.task_sessions.openenv.machine.image.kind=registry \
     +environment.task_sessions.openenv.machine.image.reference=ghcr.io/meta-pytorch/openenv-echo-env:sha-64d4b10 \
     +environment.task_sessions.openenv.machine.workdir=/app \
     '+environment.task_sessions.openenv.server_command=[python,-m,uvicorn,envs.echo_env.server.app:app,--host,127.0.0.1,--port,"8000"]'

The example data script supplies Echo and Coding rows.
Other task types require benchmark-specific data and images.
See :doc:`../tutorials/task_rollouts` for reward projection and cleanup.
