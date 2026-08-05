# Environment-variable ownership

MarinSkyRL configuration owns every environment variable the project introduces. Declare each variable once in
[`cloud/iris/env_vars.py`](../cloud/iris/env_vars.py), including its config owner and propagation scopes, then ask
`EnvVarManager` for the driver, task-runtime, Ray-worker, or inference-worker projection. Do not add literal
environment writes to launchers, trainer code, RL YAML `extra_env`, shell launch scripts, or container recipes.

Scheduler inputs, credentials, and variables owned by third-party runtimes do not need fictitious user-facing
settings. They do need an `EXTERNAL` or `SECRET` registry declaration when MarinSkyRL begins forwarding them.
Reading an inherited external variable is not a definition.

`infra/check_env_var_contract.py` records pre-existing production definition sites in
`infra/env_var_legacy_definitions.json`. CI permits those exact grandfathered counts and rejects additions. The
baseline is shrink-only: migrate a legacy site to `EnvVarManager`, then regenerate the baseline with:

```bash
uv run python infra/check_env_var_contract.py --write-baseline
```

Do not regenerate the baseline to admit a new site. A PR may bypass the check only when the user explicitly
authorizes that exception.
