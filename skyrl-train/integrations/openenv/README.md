# OpenEnv task sessions

OpenEnv uses RolloutEngine and one Shellbox machine per task.
The session starts the server in that machine and sends reset and step requests.
It returns observations and rewards. The engine closes the session before the machine.

From `skyrl-train`, prepare TaskSpecs with an explicit image and private server parameters:

```bash
uv run --project .. integrations/openenv/prepare_dummy_dataset.py \
  --output_dir "$HOME/data/openenv/echo_env" --env_name echo_env \
  --environment /path/to/environment.json --config /path/to/server.json
```

The environment file declares Linux process semantics and a prebuilt image with an `@sha256` digest.
The server file contains `server_command` and optional HTTP limits.
The preparer keeps server parameters in private verifier data.
It does not select a machine backend or turn limit.

The example entrypoint registers `OpenEnvTaskSession`.
The run config selects `environment.task_sessions.session.task_session=openenv`, machine resources, and execution limits.
See the [OpenEnv guide](../../docs/examples/openenv.rst) for the file formats and launch command.
