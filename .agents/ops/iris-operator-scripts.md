# Iris operator scripts

Run the tools under `scripts/iris/` from the repository root.

| script | purpose |
|---|---|
| `iris_ops.py` | poll authoritative lifecycle state and wait for a terminal verdict |
| `list_iris_jobs.py` | list jobs across configured clusters |
| `watch_coreweave_rl.py` | capture and summarize durable local RL evidence; see `watch-coreweave-rl.md` |
| `analyze_coreweave_rl_job.py` | analyze a captured bundle; `--local-only` makes no remote calls |
| `analyze_coreweave_rl_job_live.sh` | capture and analyze one live job |

## Runtime and state

Set `IRIS_BIN` when the operator's Iris installation differs from the helper's configured default.
Select and verify Kubernetes access as described in `coreweave.md`.

Use the lifecycle states and interpretation rules in `coreweave.md`; `iris_ops.py` owns their numeric
mapping.

```bash
python scripts/iris/iris_ops.py <job-id> --cluster <cluster> --interval 60
```

Use `--once` for automation.

## CoreWeave object storage

Read the external object-store endpoint from the current Iris cluster configuration. Obtain access
through the cluster-approved secret mechanism; do not source unrelated cloud credentials or copy
secret values into commands or docs. Use virtual-hosted bucket addressing and the region setting
declared by the cluster.

Inside Iris task pods, prefer the credentials injected by `envFrom`. Explicit `AWS_*` values can
override the correct object-store identity and redirect requests to the wrong provider.
