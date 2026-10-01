# SWE verifier and pivot behavior pilot

The seed-42 split is made by whole `metadata.instance_id` tasks. The 3,722 training
candidates are the same pool used by both frozen student profiles; validation has
256 rows over 154 tasks. The quick set contains one row for each of the first 64
validation tasks in SHA256 order (`42:<task>`), with its row chosen by
`42:<source_id>`. `infra.rl_data.pivot_pilot prepare` writes both sets and the 240
rows that need K=8 profiling for each model.

After profiling, `freeze` checks all eight repetitions and retry histories, excludes
unusable rows, and selects tool-name outcomes with one to three successes. It saves
the complete eligible pool, the frozen selected rows, a seeded count-matched random
control, and the mixed-group signal availability at G=2, 4, and 8. Those G values
are analysis-only subsamples of the original groups.

`recipes` writes six arms for each pinned student. Pass the frozen model's published
directory as `--artifact-root` and the shared split's `validation.parquet` as
`--validation-uri`. RL uses 20 optimizer updates,
64 prefixes and eight responses per update, one update epoch, no loss-token cutoff,
and KL 0.001. SFT uses a one-million-token loss budget, 64 prefixes, and seeded
reshuffling. The full 256-row evaluation runs at initialization and the prescribed
milestones. RL quick evaluation runs every two updates, with full evaluation taking
precedence at updates 10 and 20. Initialization results are cached by model,
validation split, quick subset, sampling settings, and verifier definitions.

Training responses store diagnostic grades from the tool-name, NeMo-style, and
strict argument verifiers; the configured training reward alone feeds optimization.
Required, unbounded trajectory publication writes immutable raw records plus a
Parquet grade table with one row per record. SFT records are marked as
teacher-forced. Trainer batch dumps preserve the final loss masks; consumed-token
totals and resolved advantage settings are recorded in run metrics and configuration.
Resolved YAML, manifest hashes, archive links, verifier settings, and W&B metrics
are retained with each run.

Use `infra.rl_data.pivot_pilot_report` on grade tables for full/quick curves, the
training-reward × evaluation-verifier matrix, and paired task-bootstrap intervals.
Use `infra.rl_data.pivot_report` on raw archives for exposure and first-step geometry. The initial training
stopping rules describe a behavior pilot; the report must include updates, loss and
response tokens, prefix visits, and allocated GPU-hours.
