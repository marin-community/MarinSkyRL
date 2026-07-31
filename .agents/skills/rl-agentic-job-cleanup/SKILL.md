---
name: rl-agentic-job-cleanup
description: >-
  Preserve and, when authorized, publish a terminal agentic Iris RL run: stop pending retries,
  select and validate a checkpoint, export model weights, preserve metrics and traces, verify
  destinations, and only then reclaim storage. Use for runs with an agent harness or per-trial
  artifacts; use rl-standard-job-cleanup for parquet-only RL.
---

# Clean up an agentic RL run

Complete every independent preservation step even when another step is blocked. Record which steps
completed, did not apply, or could not proceed.

## Authority

Publishing the model and companion trace dataset is part of this workflow when the campaign record
names their destinations. Routine ingress and egress within the selected CoreWeave job scope do not
require confirmation for each action. Registration is optional and requires campaign authorization.
Deletion and storage reclamation require separate explicit authority and happen only after destination
checks pass. Large or cross-region transfers remain subject to repository policy. Never invent a
namespace, registry target, base model, or campaign policy.

## Inputs

Read the terminal job chain, resolved configuration, checkpoint interval, artifact locations, base
model, publication destinations, and registration policy from current records. Cancel only exact
pending retries that belong to this run before reading mutable checkpoint directories.

## Workflow

1. **Capture the chain.** Sync every attempt's logs and durable artifacts. Preserve the tracker URL
   when one exists.
2. **Select a checkpoint.** Use `infra/rl_cleanup/parse_skyrl_metrics.py --format agentic` across the
   chronological chain to build the metric surface. The tool does not select an agentic checkpoint.
   Apply the campaign's declared selection rule only to complete saved checkpoints. If no rule is
   declared, report the valid candidates and ask for a choice. When the run ended at a known
   behavioral break, include the last checkpoint whose generating rollouts predate that boundary.
3. **Export or consolidate.** Determine the checkpoint format before converting. Follow
   `.agents/ops/checkpoint-consolidation.md` and verify that the result contains complete
   safetensors, configuration, and tokenizer metadata.
4. **Publish the model.** Assemble the metric surface, trainer log, and relevant per-step logs under
   `<staging>/training_logs/` before upload. Add a `## Training Traces` section to the model card with
   the companion Hugging Face dataset URL; this is a model-publication requirement even when trace
   publication finishes in a later pass. Redact the complete staging tree before upload:

   ```bash
   python -m infra.rl_cleanup.secret_redaction "$STAGING_ROOT"
   hf upload <namespace>/<model> "$STAGING_ROOT" --repo-type model
   hf download <namespace>/<model> --repo-type model --local-dir "$VERIFY_ROOT"
   python -m infra.rl_cleanup.secret_redaction "$VERIFY_ROOT" --check
   python -m infra.rl_cleanup.publication_checks "$VERIFY_ROOT"
   ```

   Keep staging until the fresh download passes both checks. The first check fails if credential-shaped
   text remains; the second fails unless `training_logs/` is nonempty and the card links a Hugging Face
   dataset under `## Training Traces`.
5. **Publish traces.** The monitor's `trace_jobs/` directory is an evidence sample by default; it
   mirrors only the newest 500 traces across the fleet. Rebuild the dataset from the full durable
   object-store tree instead:

   ```bash
   python infra/sync_rl_logs.py /benjaminfeuer/<job> --cluster <cluster> --dest "$TRACE_ROOT" \
     --no-ray --no-finelog --trace-jobs --trace-jobs-no-gzip
   mkdir -p "$TRACE_ROOT/trace_jobs"
   tar -xf "$TRACE_ROOT/<job>_trace_jobs.tar" -C "$TRACE_ROOT/trace_jobs"
   python -m infra.rl_cleanup.make_and_upload_trace_dataset \
     --job_dir "$TRACE_ROOT/trace_jobs" --repo_id <namespace>/<dataset> --episodes last --filter none \
     --skip_register --single_commit
   ```

   `--trace-jobs` is the source of record for this rebuild. It writes a tar plus a sync manifest;
   inspect `objects_skipped` before export. A full monitor mirror requires `--trace-sync-limit 0`,
   but remains evidence-only when its size guard skips objects. Inspect
   `trace_export_manifest.json`: `result_coverage` is dataset rows divided by source
   `result.json` files and defaults to a 95% gate. The staged tree has deterministic shard names
   and replaces stale `train-*.parquet` shards on rerun. `--single_commit` stages then replaces the
   complete remote shard set. Use `--stage-only "$STAGING_ROOT"` to retain that staged tree locally
   for inspection or offline tests; it does not write to Hugging Face. Verify the remote shard list
   and manifest before reporting the trace destination to the model-publication step.
6. **Register if authorized.** Use the campaign's schema, explicit RL training type, exact base-model
   lineage, and only records the operator owns. Stop on foreign-key or ownership ambiguity.
7. **Reclaim only if separately authorized.** Retain staging and source artifacts until all remote
   verification succeeds. Avoid expensive recursive sizing before a large delete.

## Safety rules

- A training checkpoint is not a publishable model until the export is verified.
- An automatically pushed intermediate with nested weights is not a substitute for the flat model.
- Do not let one failed upload suppress metrics, traces, or other preservation work.
- Do not use destructive upload semantics or remove remote files absent from local staging.
- Never infer model size, base lineage, or checkpoint eligibility from a run name.

## Completion record

Report terminal state, source revision, selected checkpoint and rationale, export provenance,
published model and trace destinations, registration result, preserved metrics, verification
evidence, storage reclaimed under authority, and all remaining work. For each required model artifact,
state `present`, `absent`, or `not applicable`: weights, tokenizer/configuration, `training_logs/`,
and the model-card `Training Traces` link. A model is incomplete until every applicable artifact is
`present`.
