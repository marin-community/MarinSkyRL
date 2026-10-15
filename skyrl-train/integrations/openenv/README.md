# OpenEnv task sessions

OpenEnv uses the common TaskSession interface and a Shellbox machine per rollout.
The session starts the configured server inside that machine, sends reset and step requests, and returns observations and rewards.
The engine closes the session before the machine.

The OpenEnv SDK is not a host dependency.
Supply an image and `server_command` in `environment.task_sessions.openenv`.
Select an image with the HTTP reset/step API. The included image installer lists fixed source revisions for five task types.

Supported types are Echo, Coding, OpenSpiel, Atari, SUMO, and FinRL.
FinRL requires a user-supplied image.
The example data script supplies Echo and Coding rows.

From `skyrl-train/`:

```bash
uv run --project .. integrations/openenv/prepare_dummy_dataset.py \
  --output_dir "$HOME/data/openenv/echo_env" --env_name echo_env
```

See the [OpenEnv configuration example](../../docs/examples/openenv.rst) for the machine image, server command, and training entrypoint.
