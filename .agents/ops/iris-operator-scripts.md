# Iris operator scripts

`scripts/iris/` holds the operator tooling for Iris job lifecycle and CoreWeave RL evidence
capture. Run them from the repo root; each inserts the repo root on `sys.path` itself, so a
direct invocation works.

| script | what it is for |
|---|---|
| `iris_ops.py` | Poll the authoritative job lifecycle state and exit on the terminal verdict. Shared helpers for the rest. |
| `list_iris_jobs.py` | List jobs across clusters with state and age. |
| `watch_coreweave_rl.py` | Capture the durable local evidence bundle for a running RL job. |
| `analyze_coreweave_rl_job.py` | Read that bundle and report. `--local-only` makes no remote calls. |
| `analyze_coreweave_rl_job_live.sh` | Capture and analyze in one pass against a live job. |

## Interpreter and credentials

- Python and `iris`: `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/{python,iris}`. That env
  ships `iris`, a working `kubernetes`, and `boto3`. The marin checkout's `.venv/bin/iris` cannot
  drive CoreWeave.
- `export KUBECONFIG=~/.kube/coreweave-iris` in the same shell. Without it the calls report
  misleading empty results rather than failing.
- `IRIS_BIN` overrides the iris binary if you need a different one.

## Job state

`iris_ops.py` polls state, not log content. The controller keeps terminal job records after pods
are reaped, so a job that has vanished from Kubernetes still reports its real terminal state.
Numeric states, from `lib/iris/src/iris/rpc/job.proto`:

```
0 UNSPECIFIED  1 PENDING  2 BUILDING  3 RUNNING
4 SUCCEEDED    5 FAILED   6 KILLED    7 WORKER_FAILED  8 UNSCHEDULABLE
```

Judge liveness from this and from checkpoint or trial artifacts. A raw `iris job logs` tail
interleaves ranks, and `--no-tail` returns startup lines, so neither shows whether a job is
progressing.

Example — watch a build until it reaches a terminal state:

```bash
PY=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python
export KUBECONFIG=~/.kube/coreweave-iris
$PY scripts/iris/iris_ops.py /benjaminfeuer/<job> --cluster cw-us-east-02a --interval 60
```

`--once` polls a single time and exits, which is what a monitor or cron wants.

## Provenance

Ported from the OpenThoughts-Agent checkout, where `iris_ops.py` had been renamed from
`watch_job_state.py`. MarinSkyRL does not depend on that repo; these are copies, and changes
belong here.
