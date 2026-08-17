# Environment-variable ownership

MarinSkyRL configuration owns every environment variable the project introduces. Declare each variable once in
[`cloud/iris/env_vars.py`](../cloud/iris/env_vars.py), including its config owner and propagation scopes, then ask
`EnvVarManager` for the driver, task-runtime, Ray-worker, or inference-worker projection. Do not add literal
environment writes to launchers, trainer code, RL YAML `extra_env`, shell launch scripts, or container recipes.

Scheduler inputs, credentials, and variables owned by third-party runtimes do not need fictitious user-facing
settings. They do need an `EXTERNAL` or `SECRET` registry declaration when MarinSkyRL begins forwarding them.
Reading an inherited external variable is not a definition.

`infra/check_env_var_contract.py` checks every production definition against that registry. Each entry declares
its single owner, source classification, propagation scopes, and permitted writer boundary. Marin-owned `CONFIG`
and `DERIVED` values may only be emitted by `EnvVarManager`; explicit `EXTERNAL`, `SECRET`, and build boundaries may
allow narrow Python, YAML, shell, or container writers. There is no grandfathered location-count baseline.
