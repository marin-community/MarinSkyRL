# Jupiter RL operations

Jupiter is a Slurm cluster with four GH200 GPUs per node. Keep Slurm control on the host and enter Apptainer
only for GPU worker processes. Compute nodes have no internet access, so resolve dependencies and stage models
before requesting the allocation.

## Access and job control

Use the configured `Jupiter` SSH alias when available. The direct endpoint requires IPv4:

```bash
ssh -4 -i ~/.ssh/id_ed25519_jsc "$USER@login04.jupiter.fz-juelich.de"
```

Use `squeue -u "$USER"` for live state, `sacct -j <job-id>` for terminal state, and `scancel <job-id>` only
when cancellation is authorized. Do not treat an empty result from one login host as authoritative when SSH
itself reports resource or fork errors; retry another login host.

## Secrets

The private Jupiter secrets file is `$HOME/secrets.env` on the login host. The corresponding operator-side
source is `/Users/benjaminfeuer/Documents/secrets.env`. Never print either file or copy its contents into an
ops document, launch record, or diagnostic artifact.

Agentic RL must use `DAYTONA_RL_API_KEY`; do not silently substitute the generic `DAYTONA_API_KEY`. The legacy
OpenThoughts-Agent Jupiter launcher still consumes `DAYTONA_API_KEY`, so source the private file, require the
RL-specific key, and map it explicitly immediately before invoking that launcher:

```bash
set -a
. "$HOME/secrets.env"
set +a
: "${DAYTONA_RL_API_KEY:?DAYTONA_RL_API_KEY is required for Jupiter agentic RL}"
export DAYTONA_API_KEY="$DAYTONA_RL_API_KEY"
export DC_AGENT_SECRET_ENV="$HOME/secrets.env"
```

Preflight by reporting only whether `DAYTONA_RL_API_KEY` is present. Never echo its value. After rendering,
verify that the batch job received the mapped credential before submission; a generic Daytona key can be valid
for another account while every RL sandbox request fails.

## GPFS safety

Jupiter experiment and container paths are on GPFS. Avoid recursive metadata scans such as broad `find`, `du`,
or `ls -R` under `/e/scratch` and `/e/data1`. Resolve paths from the launch record and inspect bounded directory
levels. Prefer `squeue -j <job-id> -o '%Z'` for a job's working directory.

Do not use an isolated dependency resolver on compute nodes. It cannot reach direct URL dependencies and may
replace the container's CUDA stack. Run GPU validation with the production image's Python. Keep caches and
high-churn temporary data out of shared experiment trees, and remove large per-trial trees after their durable
artifacts have been verified elsewhere.

Before cancelling a wedged distributed job, copy the required rank logs from each allocated node to a bounded
shared destination. Ray's teardown sync is not a completeness guarantee, and cancellation can destroy node-local
logs. Use one explicit node at a time with `srun --jobid=<job-id> --overlap -w <node> -N1 -n1`; never infer that
one sampled node represents the gang.

## Runtime and tests

Read [runtime.md](runtime.md) before selecting a SIF or overlay. The opt-in distributed test contracts and their
topology requirements live in
[`skyrl-train/tests/gpu/fault_injection/README.md`](../../../skyrl-train/tests/gpu/fault_injection/README.md).
The host-side controller must retain access to `scontrol` and `srun`; its remote node agents enter the policy
runtime through the explicit command prefix documented there.
