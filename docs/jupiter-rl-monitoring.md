# Jupiter RL monitoring

`scripts/iris/watch_coreweave_rl.py` combines two artifact sources in one RL
status report: automatically discovered CoreWeave Iris jobs and explicitly
selected JSC Jupiter Slurm jobs. The source-specific transports remain
separate because Iris trace selection is fleet-wide and object-store-backed,
while Jupiter artifacts live on GPFS and must be bounded per experiment.

## Metric contract

The stability cell reads canonical aliases from `infra/rl_metrics.py` and
reports four distinct signals:

- policy entropy;
- TIS served-token alignment through `generate/tis/exact_match_fraction` and
  legacy aliases;
- TIS importance-ratio diagnostics;
- policy old/new log-probability movement as mean, p99, and maximum absolute
  log delta.

The last family exposes a concentrated tail when p99 or max separates from the
mean. It is not labeled as literal top-k probability mass.

## Jupiter selection contract

Jupiter runs are opt-in:

```bash
python scripts/iris/watch_coreweave_rl.py \
  --jupiter-only \
  --jupiter-run SLURM_JOB_ID=/absolute/launcher/experiment/path
```

The experiment path is WYSIWYG: the monitor does not search scratch storage,
infer a run from a job name, or rewrite the supplied path. Slurm supplies the
launcher job name used by nested per-job artifact directories. Each run is
stored in the normal local evidence-bundle hierarchy under the `jsc-jupiter`
source name.

## GPFS bounds

The Jupiter transport only accesses known locations under the explicit
experiment root:

- `logs/` for Slurm stdout and auxiliary launcher logs;
- `ray_logs/` and `<launcher-job-name>/ray_logs/` for preserved Ray logs;
- `<launcher-job-name>/trace_jobs/` for Harbor trial artifacts.

Status-only mode remotely tails the Slurm stdout and performs no recursive
transfer. Full mode uses incremental `rsync`; the already-synced finelog is
excluded from the auxiliary log copy. Trace discovery uses a single shallow
`os.scandir` of the known `trace_jobs` directory, sorts by modification time,
and transfers only the configured newest entries. Non-log trace files retain
the existing size cap, while diagnostic log suffixes are preserved regardless
of size. Setting the Jupiter trace limit to zero is the explicit request for a
full trace sync.

SSH transport failures, listing failures, and transfer failures are surfaced
as monitor errors. They are never converted into an "artifact absent" result.

## Bundle schema

Shared fields retain their existing meanings. Jupiter adds `slurm_logs` and
`trace_selected`; it does not overload Iris `pod_logs`, `trace_started`, or
`trace_completed`. Jupiter manifests set the Iris-only `trials_uri` field to
null.

Regression tests cover metric rendering, path validation, status-only tails,
transport failure reporting, bounded trace selection, bundle schema, and the
existing CoreWeave monitor behavior.
