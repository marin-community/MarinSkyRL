# Environment-variable ownership

MarinSkyRL configuration owns every environment variable the project introduces. Declare each variable once in
[`marinskyrl/environment_contract.py`](../marinskyrl/environment_contract.py), including its config owner and propagation scopes, then ask
`EnvVarManager` for the driver, task-runtime, Ray-worker, or inference-worker projection. Do not add literal
environment writes to launchers, trainer code, RL YAML `extra_env`, shell launch scripts, or container recipes.

Scheduler inputs, credentials, and variables owned by third-party runtimes do not need fictitious user-facing
settings. They do need an `EXTERNAL` or `SECRET` registry declaration when MarinSkyRL begins forwarding them.
Reading an inherited external variable is not a definition.

Harbor tasks can select separate Daytona accounts without changing process-wide credentials:

```yaml
terminal_bench_config:
  harbor:
    environment_type: daytona
    daytona_api_key_env: DAYTONA_RL_API_KEY
    eval_daytona_api_key_env: DAYTONA_EVAL_API_KEY
```

These fields contain environment-variable names, never key values. The Iris launcher forwards the selected
credentials to task workers and fails before submission if either configured variable is missing. The existing
RL secret resolver supplies `DAYTONA_RL_API_KEY`; provide the evaluation key through the launcher's secret input.
Evaluation requests route both the task machine and its separate verifier machine through the evaluation client.
With no evaluation override, evaluation uses the training factory. Keep both values out of saved launch YAML.

`infra/check_env_var_contract.py` checks every production definition against that registry. Each entry declares
its single owner, source classification, propagation scopes, and permitted writer boundary. Marin-owned `CONFIG`
and `DERIVED` values may only be emitted by `EnvVarManager`; explicit `EXTERNAL`, `SECRET`, and build boundaries may
allow narrow Python, YAML, shell, or container writers. There is no grandfathered location-count baseline.
