---
name: rl-agentic-job-cleanup
description: >-
  Preserve AND publish a finished agentic RL run on Iris — the full checklist. Covers cancelling
  pending retries, selecting the best checkpoint by trailing-5 EMA of reward across the whole
  restart chain, flattening weights to the repo root, secret-scanning, uploading to the campaign's
  model repo, publishing the companion trace dataset, preserving metrics, optional registry entry,
  and only then reclaiming disk. Use when an agentic RL job reaches a terminal state, or when
  asked to run the RL cleanup checklist. For a parquet-only run use rl-standard-job-cleanup.
---

# Agentic RL job cleanup on Iris

Run this **end to end** after an agentic run terminates, whether it completed its step budget or
stopped early. The deliverable is a published model with weights at the repo root, a companion
trace dataset, and the metrics preserved beside the model.

Steps 8 and 9 take minutes against the hours the weight upload takes. Skipping them is how a run
ends up published but uninterpretable.

## Authority

Publishing and registering are part of this checklist. **Deleting is not.** Never remove
checkpoints, trials, remote objects, or registry rows as a side effect — reclaiming storage is a
separate, explicitly authorized step (§10), taken only after the artifact is confirmed at its
destination. If the campaign has not named a destination repo, stop and ask rather than inventing
one.

## Campaign-supplied parameters

These are properties of the campaign, not of this repo, and must come from the experiment record:

| Parameter | Role |
|---|---|
| model repo namespace | where the published weights go |
| trace dataset namespace | where rollouts go, if the campaign publishes them |
| registry target | optional — many campaigns are publish-only |
| base model id | the exact repo the run trained *from* |
| `hf_save_interval` | read from the run's resolved config, never assumed |

## Cross-cutting rules that bite

- **`hf upload`, never `hf upload-large-folder`** — the latter is a deprecated stub and deadlocks
  on Hub LFS 429s. `hf upload` is additive and safe to re-run.
- **Never call `huggingface_hub.upload_folder()` without `delete_patterns=[]`** — it removes files
  absent locally and will clobber the weights already pushed.
- **`--private` is a no-value flag.** Never pass `--private false`; default is public, so omit it.
- Wrap long uploads in `tmux`, not `nohup`.
- **A training checkpoint is not a publishable model, but it is convertible.** Confirm the export
  actually produced safetensors before promising an upload — a run can exit zero having written
  only sharded per-rank checkpoint state. When it has, consolidate it rather than reporting
  "nothing to publish"; see step 2.5. Report "nothing to publish" only when the conversion itself
  cannot proceed, and say which check stopped it.
- **A blocked step does not block the others.** Steps that need no upload — checkpoint selection,
  config capture, secret scan, metrics, trace dataset — run regardless. Finish every step that can
  finish and name the ones that could not, rather than reporting the whole cleanup as blocked.

## 0. Cancel pending retries first

A queued retry can start mid-upload and overwrite the directory being read. Find siblings of the
same run and stop them before anything else. Kill by **exact** job id: RL job names in a sweep
share long prefixes, and a prefix match will take live siblings with it.

## 1. Select the best checkpoint — trailing-5 EMA of reward, not the single-step max

Single-step max overfits one lucky batch. Use the EMA of `reward/avg_raw_reward` over a trailing-5
window.

Rules that change the answer:

- **EMA across all steps in chronological order, regardless of restarts.** A resumed run's history
  spans more than one job; collect metric lines from every link and sort by step. Never compute a
  per-link EMA.
- Standard 5-period EMA: `α = 2/(5+1) = 1/3`, `EMA_n = α·r_n + (1−α)·EMA_{n−1}`, seeded at `r_1`.
- **Never select the first saved checkpoint** — the EMA is not warmed up. Start at
  `2 × hf_save_interval`.
- Only exported steps are eligible: multiples of `hf_save_interval`, capped at the last step the
  run actually saved. Read that cap from the run's own `latest_ckpt_global_step.txt` rather than
  assuming it reached its budget.

Both the selection and the metric surface come from the ported tool
`infra/rl_cleanup/parse_skyrl_metrics.py`, which implements the rule above — chain-aware,
first-seen-wins per step, `alpha = 1/3`, first save excluded, capped at the last saved step. Do
not reimplement it inline.

```bash
python infra/rl_cleanup/parse_skyrl_metrics.py \
  --run_dir <run-dir> --save_every <hf_save_interval> <log-dir> <output-dir>
```

It prints the EMA table and the chosen step, and writes the metric surface used again in §9.
Feed it the run's own logs: fetch with `infra/sync_rl_logs.py --no-ray`, because the live log
stream is rate-limited and an unbounded tail can return a stale slice. Stage every chain link
before running it — a single mid-chain log under-covers the EMA window and can select the wrong
step.

If nothing is eligible — an early stop before the second save — the tool says so. Fall back to the
largest saved multiple and record that the selection rule did not apply.

## 2. Locate the run's metric record

Note the tracker run URL if the campaign uses one, so the published model traces back to its
curve. Not every run has one; omit rather than guess.

## 2.5. Consolidate, when the run left only sharded checkpoint state

If the selected step has no safetensors export — only per-rank sharded checkpoint files — convert
it before going further. Inspect the on-disk layout first and read the report, then convert; a
checkpoint whose sharding cannot be reassembled must fail loudly rather than produce a partial
model. Commands and the failure conditions are in
[`.agents/ops/checkpoint-consolidation.md`](../../ops/checkpoint-consolidation.md).

Record in the completion report which artifact was published: an export the run wrote itself, or
one consolidated afterwards. They are not interchangeable provenance.

## 3. Flatten the model files to the upload directory ROOT

Hub model files must sit at the base of the uploaded directory, not nested under `policy/`. Stage
a clean directory, copy the export's contents to its root, then confirm the safetensors, config,
and tokenizer files are all at top level.

## 4. Copy the launch configuration in beside the weights

The resolved config the run actually used, so the artifact is reproducible without the cluster.

## 5. Scan for secrets before uploading

The campaign's own artifacts carry a known live credential; match its shape explicitly rather than
relying on a scan assembled from provider key prefixes, which does not catch it. The pattern and
where it appears are in [`.agents/ops/coreweave.md`](../../ops/coreweave.md). When no scanner binary
is installed, say so in the report and state which patterns you did match.

The Hub scans after upload; catch it first. Run a secrets scanner over the staged directory and
over any logs or traces being published, or fall back to a pattern grep for provider key shapes
and JWTs. Remove or redact anything found before proceeding.

## 6. Upload the weights

The canonical sequence — export, verify, stage flat, scan, upload, verify — together with the
destination namespace and the exact commands, lives in
[`.agents/ops/checkpoint-consolidation.md`](../../ops/checkpoint-consolidation.md). Follow it there
rather than reconstructing it; the upload flags in particular have sharp edges that have bitten
before.

Two rules that are methodology, not recipe, and hold wherever this runs:

- **Derive the model size from the exported weights**, by summing tensor shapes or reading the
  safetensors index — never from the run name or the base-model name, which are frequently
  misleading about what was actually trained.
- **An auto-pushed repo is not the deliverable.** Some trainers push intermediates with weights
  nested under a step directory, which is not `from_pretrained`-able. Publish the manually
  flattened export and treat anything auto-pushed as a duplicate to reconcile, not as done.

## 7. Register in the campaign's registry — OPTIONAL

Many campaigns are publish-only. **Register only if the experiment record says the series is
registerable**, using the campaign's own tool and schema. If it is ambiguous, stop and surface it:
an unwanted registry row is harder to retract than to avoid.

Two rules whenever a registry is in play:

- **Set the training type explicitly.** Registration tooling commonly defaults to supervised
  fine-tuning, which silently mislabels an RL artifact.
- **Cross-user foreign-key safety before any delete.** If another user's rows reference the row you
  are about to remove, **stop**: leave the duplicate and surface the conflict. One row of noise is
  far cheaper than breaking someone else's evaluations. Restrict every write to rows you own.

Verify the base-model id carefully — the exact repo the run trained *from*, cross-checked against
the resolved config's policy model path rather than inferred from the job name. Getting it wrong
corrupts the lineage used for downstream size and improvement analysis.

## 8. Publish the companion trace dataset

Agentic runs produce rollouts that carry most of their interpretive value. Publish them with the
ported tool `infra/rl_cleanup/make_and_upload_trace_dataset.py`:

```bash
python -m infra.rl_cleanup.make_and_upload_trace_dataset \\
  --job_dir <job-dir> --repo_id <dataset-namespace>/<job> --episodes last --skip_register
```

It reads the inner job subdirectory where the harness writes `trace_jobs/`, and applies the
dataset hygiene in `infra/rl_cleanup/trace_dataset_hygiene.py` — surrogate stripping, bash-warning
sanitisation, ShareGPT conformance. Those are load-bearing: without them PyArrow rejects the
dataset. Do not bypass them.

- **Never subsample or cap.** Slow is acceptable; an incomplete dataset is not.
- RL trace datasets are often not registered even when the model is — follow the campaign's
  convention rather than defaulting.
- Trace publishers can buffer the entire dataset before pushing, so peak memory scales with trial
  count. If a large run's upload dies on memory, report that as the bug; do not "fix" it by
  sampling.

Then add a short **Training Traces** section to the model card linking the dataset, appending to an
existing card rather than overwriting it.

## 9. Preserve the metrics beside the model

Reuse the §1 run of `infra/rl_cleanup/parse_skyrl_metrics.py` — it already emitted the metric
surface. Copy that output plus the trainer log and per-step logs into a `training_logs/` directory
inside the upload dir, then re-upload additively. Where a campaign has no metrics
tracker, this is the only durable record of the run's learning curve.

## 10. Reclaim disk — last, and only when authorized

Only after every step above is confirmed at its destination. Detach a large delete; never size the
directory with a recursive scan first. Keep the staging directory until the published repo is
verified to list both the weights and `training_logs/`.

## Completion record

Report the terminal state, the selected step and why the EMA chose it, the published repo and
dataset, whether registration applied, preserved artifact locations, storage totals, and anything
left pending. State plainly which steps did not apply and which could not complete — a checklist
reported as done when a step was silently impossible is worse than one reported as blocked.
