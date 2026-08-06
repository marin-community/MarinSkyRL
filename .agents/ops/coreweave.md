# CoreWeave Iris operations

This runbook owns mutable cluster access and job-operation facts. Confirm them against the current
Marin Iris cluster configuration before acting.

## Clusters

| Iris cluster | common workload | architecture |
|---|---|---|
| `cw-us-east-02a` | H100 RL and exports | amd64 |
| `cw-rno2a` | H100 RL and image builds | amd64 |
| `cw-us-east-08a` | GB200 workloads and arm64 image builds | arm64 GPU nodes |

Resolve the kubeconfig and CoreWeave-capable Iris/Python environment at execution time. Verify the
active Kubernetes context against the selected Marin Iris cluster configuration. There is no
SSH/login-node workflow; submit and inspect through Iris and Kubernetes in namespace `iris`.

Read allocatable resources, queue state, priority policy, object-store endpoints, and secret
injection from `marin/lib/iris/config/<cluster>.yaml` and the live cluster. Do not rely on remembered
capacity.

## Job state

Controller states are authoritative: pending, building, running, succeeded, failed, killed,
worker-failed, and unschedulable.

```bash
iris --cluster=<cluster> query \
  "SELECT job_id, state, submitted_at_ms, finished_at_ms FROM jobs WHERE job_id='<job-id>'" -f csv
iris --cluster=<cluster> job list --prefix <job-id>
iris --cluster=<cluster> job summary <job-id>
```

- Poll state; do not wait for a terminal log string.
- A running state does not prove training progress. Require advancing phases, steps, or artifacts.
- A federated job may be absent from a target cluster until delegation. Preserve the launcher's exact
  job ID and check federation before resubmitting.
- If a previously observed row disappears after retention and no pods remain, treat the job as
  terminal but recover success/failure from durable artifacts.
- Retry task IDs may include an attempt suffix. Strip it before deriving ranks.

## Logs and evidence

Capture once, then analyze local files:

```bash
python infra/sync_rl_logs.py <job-id> --cluster <cluster> --dest <directory> [--trace-jobs]
```

Read the aggregate `finelog.log` and per-actor files under `ray_session_logs/`. Bound live log
queries by time or line count. Use `.agents/ops/rl-diagnostics.md` for interpretation.

For process stacks on a live suspect, locate `py-spy` inside the task container and attach to the
specific trainer, policy worker, or inference process. A single collective stack or watchdog line
does not prove a wedge; reconcile controller state, fresh logs, restarts, and all affected ranks.

## Capacity and GPU probes

`Ready` nodes are not necessarily free. Check the cluster queue and count whole nodes whose bound
pod requests leave enough GPUs, CPU, and memory for the gang. Multi-node jobs are admitted as a
unit; scheduling-gated pods before admission are normal.

```bash
kubectl get clusterqueue
kubectl -n iris get pods -o wide
kubectl -n iris exec <pod> -c task -- nvidia-smi \
  --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv
```

Use `rl-diagnostics.md` to interpret the captured GPU and engine state.

## Networking

Use the cluster-portable NCCL interface exclusion configured by the launcher; do not hardcode a
cluster's Ethernet interface. Healthy multi-node initialization should select InfiniBand and GPU
Direct RDMA. A missing optional NCCL plugin message can be benign; interface-selection or watchdog
failures are not.

## Agentic sandboxes

- Use the launcher-supported secret indirection for the dedicated RL sandbox provider account.
- Read current snapshot quotas and cleanup policy from the provider and launcher before submission.
- Quota exhaustion can make every trial fail before scoring; inspect an actual trial exception.
- Hard termination can orphan in-flight sandboxes. Reap only those proven idle and only with
  authority.

## Secret-bearing artifacts

Agent ingress URLs can embed capability JWTs in resolved configuration, trainer logs, and trial
metadata. Before publishing any derived artifact, scan for JWT-shaped values inside URLs as well as
named provider keys. Treat discovered tokens as live regardless of apparent age and redact them.

## Trial artifacts

Agentic trials are stored under the run's durable `trace_jobs/` prefix and commonly contain a start
configuration, finalized result, agent/verifier outputs, and an exception file. Discover the actual
run prefix from resolved metadata; do not assume the literal configured path survived launcher
rewrites.

Aggregate large trial sets inside a task pod and transfer summaries rather than syncing hundreds of
thousands of small objects to a laptop. Compare timezone-aware object timestamps in UTC; CLI display
may use local time.

## Runtime environments

The launcher checks out its own immutable MarinSkyRL commit in the standard Iris task image and runs
`uv sync --frozen` with the dependency profile selected by `trainer.strategy`. Verify the commit and
profile in the launch banner and the checkout revision in task setup logs. Architecture-specific
Python wheels are selected by the upstream lock; there is no RL image override.

## Guardrails

- Never stop a running job, restart a cluster, delete artifacts, or mutate credentials without
  explicit authority.
- Use exact job IDs for lifecycle actions. Dry-run any prefix operation and inspect every match.
- Read retry policy and concurrency limits from the current launcher and campaign record.
- Iris environment flags take separate key and value arguments.
