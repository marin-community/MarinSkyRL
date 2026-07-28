# CoreWeave clusters — access, state, and babysitting primitives

Cluster-access facts for the two CoreWeave GPU clusters that run MarinSkyRL RL jobs. Skills
point here; this file owns the commands and constants. Update this file when a fact changes —
do not copy its contents into skills.

## Clusters and access

| | `cw-us-east-02a` (East) | `cw-rno2a` (Reno) |
|---|---|---|
| Fleet | ~32 usable 8×H100-80GB IB nodes (~256 GPU) | 64× 8×H100-80GB IB nodes (512 GPU), pool pinned warm |
| KUBECONFIG | `~/.kube/coreweave-iris` | `~/.kube/coreweave-iris` |
| kube context | `marin-gpu_US-EAST-02A` | `marin-rn02a_RNO2A` |
| Iris cluster config | `marin/lib/iris/config/cw-us-east-02a.yaml` | `marin/lib/iris/config/cw-rno2a.yaml` |
| Object store | `s3://marin-us-east-02a` via `https://cwobject.com` (external) / `http://cwlota.com` (in-cluster) | same bucket, same endpoints |

- No SSH and no login node. Everything goes through the `iris` SDK over a controller tunnel.
- `iris` binary: `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris` (the CoreWeave-capable
  install). Export the cluster's KUBECONFIG in the same shell for every `iris`/`kubectl` call.
- Namespace `iris` holds the controller, finelog server, and all task pods.
- Whole-node-exclusive scheduling: node allocatable ≈ 128 CPU / ~2014 GiB / 8 GPU. Launcher
  defaults: `--cpu 48`, `--memory 1400GB` (higher risks a Kueue topology-fit stall — on an
  admission stall, lower `--memory`, never raise a cap), `--gpus_per_node 8`.
- Multi-node jobs are Kueue gang-scheduled (`leafgroup` coscheduling, all-or-nothing).
  `SchedulingGated` pods before admission are normal.

## Priority bands

`--priority {production, interactive, batch}`; higher bands preempt lower. Batch has no node
cap and is fully preemptible. **Interactive is allowed only on `cw-rno2a`, and only when the
rno2a queue is full (a batch submission would sit pending behind existing load). Never use
interactive on any other CoreWeave cluster.** Default to batch.

## Job state — poll it, never grep for terminal log strings

JobState codes: 1 pending, 2 building, 3 running, 4 succeeded, 5 failed, 6 killed,
7 worker_failed, 8 unschedulable.

```bash
IRIS=/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris
$IRIS --cluster=<cluster> query \
  "SELECT job_id, state, submitted_at_ms, finished_at_ms FROM jobs WHERE job_id='<job_id>'" -f csv
$IRIS --cluster=<cluster> job list --prefix <job_id>     # includes pending_reason and children
$IRIS --cluster=<cluster> job summary <job_id>           # per-task state/exit/duration/peak-mem
```

- A clean kill, eviction, or preemption often emits no terminal log line and reaps the pods —
  a log-content watch then hangs forever on a dead job. State is authoritative.
- The `jobs` table is pruned (terminal rows deleted after retention; federated mirror rows can
  vanish as soon as the peer tombstones them). **Absence-after-existence is terminal**, not
  "still running": if the row is gone AND `kubectl -n iris get pods | grep <job>` shows 0 pods,
  the job is over. Absence proves the job ended, not that it succeeded — for pass/fail read
  the state inside the retention window or check the job's durable artifact (checkpoint dir,
  HF repo, s3 prefix). This rule applies only to a job **previously observed** in the table.
- A job submitted through the marin meta-scheduler does not appear in the target cluster's
  `jobs` table until it is **delegated** to that cluster. `submitted_at_ms` records the
  delegation time; the job name carries the launcher's mint timestamp, so the two can differ
  by hours (a job named `…-20260726-101418-bb6bff` at 10:14 UTC was absent from the table at
  10:47 and appeared with `submitted_at_ms` 14:33 UTC when capacity freed). Absence for a
  never-observed job therefore means "not yet delegated, or pruned, or never launched" — it
  does not prove the launch failed. Read the resolved job id from the launcher's own output
  and poll by name across the delegation window; resubmitting on absence alone risks a
  double-run against one checkpoint lineage.
- On a retried or preempted attempt `IRIS_TASK_ID` gains a `:N` suffix (`/user/job/0:2`);
  rank parsing must strip it (`.rsplit('/', 1)[-1].split(':', 1)[0]`) or it crashes the
  moment any rank is retried.
- `RUNNING ≠ stepping.` A post-bring-up trainer wedge stays state 3 with a benign heartbeat as
  the last log line. Liveness for a trainer is forward advancement (a fresh training step /
  advancing phase timers), never engine bring-up completion.

## Log capture

Sync logs local once, then analyze from files (repeated live `iris job logs` greps are slow,
unbounded, and unreproducible):

```bash
set -a; source /Users/benjaminfeuer/Documents/secrets.env; set +a
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python infra/sync_rl_logs.py \
    /benjaminfeuer/<job> --cluster <cluster> [--run run-<ts>] [--dest DIR] [--trace-jobs]
# → <slug>-<run>/finelog.log  +  <slug>-<run>/ray_session_logs/  (+ optional trace_jobs archive)
```

- `finelog.log` is the aggregated controller/job stream — terminating exceptions, NCCL
  timeouts, `Worker rank N received signal` live here. Read it first.
- `ray_session_logs/` holds per-actor `worker-*.out/.err` — vLLM `EngineCore` fatals, Ray actor
  deaths, per-rank NCCL `Init COMPLETE`. Grep `worker-*`, not `python-core-*.log` (traceback-less
  C++ core logs). A death with a clean finelog can still have its killer here; read both before
  any verdict.
- Bounded live fetch when needed: `iris --cluster=<cluster> job logs <job_id> --since-seconds N --no-tail`.

## Capacity and headroom

`kubectl get nodes` shows Ready, not free. The two authoritative signals:

```bash
kubectl get clusterqueue                      # PENDING WORKLOADS 0 = no admission backlog
# free whole nodes = per-node (allocatable GPU − bound-pod GPU requests|limits), count nodes with ≥8 free
```

Submit an N-node gang only when pending workloads are 0 and free whole nodes ≥ N. "The queue is
full" (for the interactive-priority rule above) means this check fails for a batch submission.

## GPU poll (Gate B input)

```bash
kubectl -n iris get pods -o wide | grep -aiE '<job-substr>'
kubectl -n iris exec <pod> -c task -- nvidia-smi \
  --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv
```

Separate policy-mesh ranks from vLLM engine ranks before interpreting utilization (see
`rl-diagnostics.md`). Point-in-time SM-util is a trap; use the saturation tuple.

## py-spy on a wedged rank (capture before any kill)

Works on CoreWeave (`task` container has `CAP_SYS_PTRACE`). py-spy is not on `$PATH`; resolve
from the uv cache:

```bash
kubectl -n iris exec <pod> -c task -- bash -lc \
  'P=$(find /root/.cache/uv /uv/cache -name py-spy -type f 2>/dev/null | head -1); "$P" dump --pid <PID>'
# PIDs: ps -eo pid,args | grep -aiE 'skyrl_entrypoint|PolicyWorker|EngineCore'
```

A py-spy barrier snapshot plus a lone NCCL watchdog log line is not a wedge verdict — a real
tripped watchdog aborts the process. Require pod-restarts 0 + stalled fresh logs on all nodes +
a reconciled timeline before calling wedge.

## NCCL / networking

- `NCCL_SOCKET_IFNAME` must be the cluster-portable exclusion list
  `"^ibs,ibp,lo,docker,veth,cilium,lxc"` — a hardcoded Ethernet PF (e.g. East's `enp157s0np0`)
  does not exist on the other cluster and kills the bootstrap (`no socket interface found`);
  omitting it entirely on rno2a wedges the gang silently (raylets never connect).
- Healthy IB signal at launch (`NCCL_DEBUG=INFO`): `NET/IB : Using [0]mlx5_0:1/IB` +
  `GPU Direct RDMA Enabled`. The `Could not find libnccl-net.so` line is benign.
- Do not set the NCCL disable knobs (`NCCL_P2P_DISABLE` etc.); defaults give NVLink intra-node
  + IB inter-node.

## Daytona (agentic rollout sandboxes)

- RL always runs on the dedicated RL org: launch with `--daytona-api-key-env DAYTONA_RL_API_KEY`
  (a pre-launch `export DAYTONA_API_KEY=…` is clobbered — the launcher re-sources `secrets.env`,
  file overrides shell). Verify in-pod: `printenv DAYTONA_API_KEY | sha1sum`.
- The RL org enforces a **40-snapshot quota**, and harbor mints one `harbor__*` env snapshot
  per trial (`auto_snapshot=true`). Over quota, snapshot creation fails and harbor falls
  through to a declarative sandbox build that this org forbids: every trial dies unscored
  with `DaytonaValidationError` and the job trains on all-zero rewards — an infra
  fingerprint, not a model or dataset problem.
- The launcher purges `harbor__*` snapshots idle past 2 h before every launch
  (`_purge_stale_daytona_snapshots`), which keeps enough headroom for harbor's worker-side
  minting to self-heal; no manual purge is needed on the launch path.
- A hard job kill orphans in-flight sandboxes (the coordinator dies before teardown). After a
  kill, reap idle sandboxes once they cross the idle threshold (~1–2 h later), never immediately
  — an aggressive reap kills other jobs' active trials.

## Known secret in run artifacts: the ingress capability JWT

`terminal_bench_config.agent_api_base` is a minted ingress URL that **embeds a live iris
capability JWT**. Nothing redacts it. It reaches at least two artifacts a cleanup stages:

- the run's resolved config, and
- the trainer log (the endpoint is echoed at harbor spawn).

It is also in **every trial artifact** — measured at 21,619 of 21,619 `result.json` files in one run,
at `config.agent.kwargs.api_base`. Anything derived from raw trials must be scanned before it leaves
the cluster. The trace exporter keeps it under `raw_metadata` and does not appear to emit it, but
scan the built dataset before publishing rather than relying on that.

Any secret scan run before publishing MUST match a JWT shape (`eyJ...\.eyJ...`) inside a URL, not
only bare `TOKEN=` assignments — this one is neither a named-credential assignment nor a
provider-prefixed key, so a scan built only from those patterns misses it. It has been found in
staged material once already.

The token is 24h-lived, so an old artifact is usually inert, but treat it as live: expiry is not a
control we own, and a published repo outlives any reasoning about it. Redact before staging.

## Trial artifacts (trace_jobs)

- Trials live at `s3://marin-us-east-02a/iris/<run-name>/trace_jobs/<trial>/` with
  `config.json` (written at trial start), `result.json` (written at finalize; carries the
  reward and `TimingInfo`), `agent/`, `verifier/`, and `exception.txt` on error.
  `<run-name>` comes from `trainer.run_name`, not the config's literal `trials_dir` (the
  coordinator rewrites it) — discover it by listing `iris/` and matching the job name.
- Aggregate in-pod and transfer aggregates only — never sync raw trials to the Mac
  (hundreds of thousands of small objects). In-pod access: run the aggregation with
  **`/opt/openthoughts/envs/rl/bin/python`** — the `/app/.venv` bundle python has no boto3, so a
  script run with it fails on import rather than on credentials. Use boto3 with
  `endpoint_url="http://cwlota.com"` and `Config(s3={"addressing_style": "virtual"})`; one
  paginated `list_objects_v2` over `.../trace_jobs/`, newest ~200 (steady state) / ~500
  (error tails) by `LastModified`. What the fields mean: `rl-diagnostics.md` §Per-trial
  duty cycle.
- **`LastModified` is UTC in boto3 and LOCAL time in `aws s3 ls`.** The AWS CLI renders the
  timestamp in the machine's timezone; boto3 returns a tz-aware UTC datetime. The same object
  read both ways at one instant:

  ```
  aws s3 ls  ->  2026-07-27 04:25:23              (local, EDT)
  boto3      ->  2026-07-27T08:25:37+00:00        (UTC)
  ```

  This is a client-side rendering difference, NOT a property of the endpoint. Both directions
  have already caused errors on this fleet: reading `aws s3 ls` output as UTC makes a live job
  look four hours dead, and "correcting" a boto3 timestamp by +4 h puts objects in the future.
  Prefer boto3 and compare against `datetime.now(timezone.utc)`; if you use the CLI, compare
  against local `date`, not `date -u`.

## Images

- The RL image is pinned by immutable digest in `cloud/iris/launch_rl_iris.py` (never the floating
  tag — it stale-caches under `imagePullPolicy: IfNotPresent`). There are TWO pins, and the
  launcher picks between them from the config's `trainer.strategy`:
  `DEFAULT_RL_MEGATRON_DOCKER_IMAGE` for `megatron`, `DEFAULT_RL_DOCKER_IMAGE` otherwise. The
  plain image has no megatron package, so a megatron config on it dies at import. Pass
  `--task-image` only to override that choice; the launch banner echoes the resolved digest, and
  `kubectl -n iris get pod <pod> -o jsonpath='{.status.containerStatuses[*].imageID}'` confirms
  what the pods actually run.
- First-party source ships via the `/app` workspace bundle at launch; only baked/compiled
  contents (vLLM fork, flash-attn, torch/CUDA, locked deps, harbor) require an image rebuild —
  see the `build-gpu-rl-image-iris` skill.
- Every image layer must stay under ~8 GB or cold pulls fail (`ImagePullBackOff` with
  restart-from-zero blob reads); verify layer sizes before swapping a live job onto a new image.

## Standing guardrails

- Never kill a RUNNING job or restart/bounce a cluster without explicit permission.
- `iris job kill|stop <id>...` matches **exactly** by default and accepts several ids at
  once. `--prefix` opts into prefix matching, which also terminates every job whose id
  extends the one you named (`abb-qwen35-35b` takes `-v2` … `-v5` with it) — run `--dry-run`
  first on any `--prefix` kill, and never name a relaunch as an extension of a live job's id.
- Keep at most 6 running RL jobs total (Daytona capacity, not per-cluster).
- `--max-retries ≥ 1` on gangs (transient HF weight-resolution flakes SIGKILL a gang at
  `max_retries=0`).
- Iris env flags are two-argument: `-e KEY VALUE`, not `-e KEY=VALUE`.
