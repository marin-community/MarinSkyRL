# Harbor task preparation

Prepare Harbor packages before the training command. From the repository root:

```bash
uv run --frozen python -m marinskyrl.harbor_preparation \
  --input /path/to/harbor-packages \
  --output-dir /path/to/prepared-tasks \
  --execution-output /path/to/execution.json
```

The command prints the TaskSpec Parquet path. It also writes per-source machine
resources, users, network policies, and deadlines to `execution.json`.
Merge its `environment.task_sessions.sources` mapping into the same path in the run config.
The training command does not read the execution JSON automatically.
Select the Shellbox backend in the run config.

The importer accepts single-stage tasks with prebuilt, digest-pinned images and
separate verifier environments. It rejects shared grading, image builds, and multi-stage packages.

Use `skyrl_train.entrypoints.main_base` for training or
`skyrl_train.entrypoints.task_generate` for generation without policy updates.
The trainer receives prepared TaskSpecs, not Harbor directories.
See [task rollouts](../../docs/tutorials/task_rollouts.rst) for runtime settings.
