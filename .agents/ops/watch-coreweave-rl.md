# RL watcher and local evidence bundles

`scripts/iris/watch_coreweave_rl.py` is the standard first probe for CoreWeave Iris RL jobs and
explicitly selected JSC Jupiter RL jobs. It is read-only: it discovers jobs, synchronizes bounded
evidence into stable local directories, extracts the latest training metrics, and writes a fleet
report.

Inspect the local bundle before querying a cluster. Refresh the bundle first when its manifest is
older than the decision being made. Use live cluster probes only for evidence that is unavailable
locally or disappears when a job exits, such as current GPU state and process stacks.

## Invoke the watcher

Run from the MarinSkyRL repository root with the frozen environment:

```bash
uv run --frozen python scripts/iris/watch_coreweave_rl.py --user <iris-user>
```

The default sweep covers RL jobs submitted in the last 24 hours across every configured CoreWeave
GPU cluster. It includes active jobs plus succeeded and failed jobs in that window. A full sweep
downloads the complete available Finelog and, for active Iris jobs, current pod stdout, Ray/vLLM
logs, and a bounded set of agentic traces. It also reports the selected user's Iris budget.

Use a Finelog-only sweep for a quick fleet survey:

```bash
uv run --frozen python scripts/iris/watch_coreweave_rl.py \
  --user <iris-user> \
  --status-only
```

Use a full, targeted sync before a deep dive:

```bash
uv run --frozen python scripts/iris/watch_coreweave_rl.py \
  --user <iris-user> \
  --hours 0 \
  --filter 'job=^/<iris-user>/<job-name>$'
```

Repeat `--filter KEY=REGEX` to combine case-insensitive filters. Available keys are `cluster`, `job`,
`name`, `dataset`, `type`, `state`, `submitted`, and `duration`. All filters must match.

Jupiter runs are never discovered automatically. Supply the Slurm job ID and canonical remote
experiment directory explicitly. This avoids broad GPFS scans:

```bash
uv run --frozen python scripts/iris/watch_coreweave_rl.py \
  --jupiter-only \
  --jupiter-run <slurm-job-id>=/absolute/path/to/experiment \
  --status-only
```

Omit `--status-only` for the bounded full Jupiter sync. Repeat `--jupiter-run` to include more than
one run. The configured SSH alias defaults to `Jupiter`; override it with `--jupiter-host` when
needed.

## Synchronization bounds

- `--status-only` refreshes the current Finelog tail and skips pod, Ray, and trace transfer.
- A full active-Iris sweep selects the 500 most recently modified trace directories across the
  whole discovered fleet. Selection uses object-store `LastModified`, not trace names.
- `--trace-sync-limit 0` requests every Iris trace. Treat this as a potentially large transfer.
- A full Jupiter sync selects the 20 newest trace directories per explicit run and performs one
  shallow GPFS listing. `--jupiter-trace-sync-limit 0` requests every trace for that run.
- Non-log trace objects larger than 100 MiB are skipped by default. Diagnostic log files are still
  copied. Change the bound with `--max-non-log-bytes`; zero disables it.
- Succeeded and failed Iris jobs receive a Finelog sync only. Their pods may already be gone, and
  they do not enter the active-fleet trace selection.

Use the default bounds for routine supervision. Narrow the job set before increasing them.

## Local layout

The default bundle root is `~/Documents/iris-job-bundles`. Override it with `--bundle-root`.

```text
~/Documents/iris-job-bundles/
├── reports/rl/
│   ├── latest.md
│   ├── latest.json
│   ├── latest-errors.md
│   ├── <timestamp>.md
│   └── <timestamp>.errors.md
└── jobs/
    ├── <iris-cluster>/<iris-user>/<job-name>/
    │   ├── manifest.json
    │   ├── finelog.log
    │   ├── finelog.stderr
    │   ├── pod_logs/
    │   ├── ray_vllm_logs/
    │   └── trace_jobs/
    │       ├── sync_selection.json
    │       └── skipped_objects.json
    └── jsc-jupiter/slurm/<slurm-job-id>/
        ├── manifest.json
        ├── finelog.log
        ├── slurm_logs/
        ├── ray_logs/
        └── trace_jobs/
```

Repeated sweeps refresh the same per-job directory. They do not create a new job directory for an
Iris retry or delete traces selected by an earlier sweep. Before relying on a file, read
`manifest.json`: verify `synced_at`, the job identity, and the status for each artifact class. A
recent `--status-only` run can coexist with older `pod_logs/`, `ray_vllm_logs/`, or `trace_jobs/`.

`reports/rl/latest.md` is the stable human-readable fleet report. `latest.json` records each local
job directory, artifact status, budget, report path, and error-report path for automation. Every
sweep also writes timestamped Markdown report and error files. Read `latest-errors.md` whenever the
report's error count is nonzero; sync and parse failures are kept out of the compact status cells.

## Report fields

The watcher reports:

- job backend, cluster, and short name;
- dataset extracted from the submitted command;
- controller state, with `awaiting placement` when an active Iris task has not reached a node;
- latest training step and configured total when present;
- reward, policy loss, and gradient norm from the latest parsed training record;
- trace selection and transfer status;
- entropy, TIS alignment or ratio diagnostics, and the mean/p99/maximum absolute token log-probability
  shift;
- per-cluster Iris budget and a separate monitor-error count.

The table is a survey, not a health verdict. `running` does not prove progress, a missing
`trace_jobs/` directory is normal for standard RL, and an unavailable metric can mean bring-up,
buffering, disabled instrumentation, or a parse failure. Interpret the synchronized evidence with
`rl-diagnostics.md`.

Render watcher data in user-facing Markdown as a GitHub-flavored pipe table. This example uses the
same columns as the watcher:

| Job | Dataset | State | Step | Reward | Policy Loss | Grad Norm | Traces | Trend |
|---|---|---|---:|---:|---:|---:|---|---|
| `cw-rno2a/rl-example` | `s3://bucket/train.parquet` | running | 27/80 | -0.0933 | -2.59e-09 | 0.0569 | newest 4/20 selected | entropy=1.143; TIS exact=0.99; TIS \|log r\|=0.012; token \|Δlog p\| μ/p99/max=0.001/0.04/0.65 |

## Escalate to live probes

Start with `latest.md`, `latest-errors.md`, and the target job's `manifest.json` and `finelog.log`.
Then inspect local pod, Ray/vLLM, Slurm, and trace files appropriate to the backend. Refresh with a
targeted full sync if the required artifact was not requested or is stale.

Query Iris, Kubernetes, Slurm, or a live process only when:

- the watcher reports an unavailable or contradictory artifact;
- the decision depends on current placement, GPU state, queue state, or process stacks;
- the job may be silently stalled and live rank evidence is required; or
- the required artifact is not durable and will disappear at termination.

Use `coreweave.md` for live Iris and Kubernetes access. Use `jupiter/README.md` for bounded Jupiter
operations. Use `rl-diagnostics.md` for signal interpretation.
