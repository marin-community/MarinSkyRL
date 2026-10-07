OpenEnv task sessions
=====================

``OpenEnvTaskSession`` implements the common task interface.
Shellbox creates one machine for each rollout from a prebuilt, digest-pinned image.
The session starts the configured OpenEnv server in that machine.
HTTP reset and step requests run inside the machine.

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

   from skyrl_gym.openenv_tasks import OpenEnvTaskSession
   from skyrl_train.entrypoints.main_base import BasePPOExp

   experiment = BasePPOExp(cfg, sessions={"openenv": OpenEnvTaskSession})

Machine configuration
---------------------

Save this configuration as ``/absolute/path/openenv.yaml``.
Replace ``IMAGE@sha256:DIGEST`` with the full digest-pinned reference for your prepared image:

.. code-block:: yaml

   defaults:
     - ppo_base_config
     - _self_

   environment:
     task_sessions:
       openenv:
         machine:
           requirements:
             docker_image: IMAGE@sha256:DIGEST
             working_directory: /app
           runtime:
             backend: docker
             network: deny
             cpus: 1
             memory_mb: 1024
             storage_mb: null
             gpus: 0
             user: null
             startup_timeout: 600
             cleanup_timeout: null
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

Session limits come from ``environment.task_sessions.session``.
An ``openenv.session`` block can override them.
The session reads its turn limit from the lowered record.

The selected image must supply the HTTP API that this integration calls.
The server accepts ``POST /reset`` with an empty object.
For ``<action>hello</action>``, the Echo request to ``POST /step`` is:

.. code-block:: json

   {"action": {"message": "hello"}, "timeout_s": 15}

The other action payloads are:

- Coding: ``{"code": "print(42)"}``.
- OpenSpiel: ``{"action_id": 2, "game_name": "catch", "game_params": {}}``. The row can override ``game_name``.
- Atari: ``{"action_id": 2, "game_name": "pong", "obs_type": "rgb", "full_action_space": false}``.
- SUMO: ``{"phase_id": 2, "ts_id": "0"}``.
- FinRL: ``{"actions": [0.1, -0.2]}``.

Each response contains an ``observation`` object, a numeric or null ``reward``, and a boolean ``done``.
The session converts a null reward to zero.
The final grade is the mean of completed turn rewards, including invalid actions and null rewards.
``GET /health`` returns a successful HTTP status before the server startup deadline.
The host does not require the OpenEnv Python SDK or exposed container ports.

For mixed task types, ``environment.task_sessions.openenv.machines`` maps each row's ``env_name`` to its machine configuration.
Each entry contains ``requirements`` and ``runtime``, as the ``machine`` block above shows.
The selected entry overrides the common ``machine`` block.
The server command remains common to all entries.
Different applications therefore require prepared images with the same server entrypoint, for example:

.. code-block:: yaml

   environment:
     task_sessions:
       openenv:
         server_command: [/app/start-server]
         machines:
           echo_env:
             requirements:
               docker_image: ECHO_IMAGE@sha256:DIGEST
               working_directory: /app
             runtime: &openenv_runtime
               backend: docker
               network: deny
               cpus: 1
               memory_mb: 1024
               storage_mb: null
               gpus: 0
               user: null
               startup_timeout: 600
               cleanup_timeout: null
           coding_env:
             requirements:
               docker_image: CODING_IMAGE@sha256:DIGEST
               working_directory: /app
             runtime: *openenv_runtime

This row selects the ``echo_env`` entry:

.. code-block:: python

   row = {
       "prompt": [{"role": "user", "content": "Return the action text."}],
       "env_class": "openenv",
       "env_name": "echo_env",
   }

Source conversion puts selected requirements in ``TaskSpec.environment_requirements``.
It puts the machine selection in ``LoweredTaskSpec.runtime.task_machine``.
The remaining settings, including the server command, stay in the private verifier payload.

Prepare and launch
------------------

From ``skyrl-train/``, prepare the source rows:

.. code-block:: bash

   uv run --project .. integrations/openenv/prepare_dummy_dataset.py \
     --output_dir "$HOME/data/openenv" --env_name echo_env

Then run the example with the machine configuration above:

.. code-block:: bash

   bash integrations/openenv/run_openenv.sh \
     --config-dir /absolute/path --config-name openenv

The example data script supplies Echo and Coding rows.
Other task types require benchmark-specific data and prebuilt images.
See :doc:`../tutorials/task_rollouts` for reward projection and cleanup.
