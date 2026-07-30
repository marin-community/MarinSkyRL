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
names their destinations. Registration is optional and requires campaign authorization. Deletion and
storage reclamation require separate explicit authority and happen only after destination checks pass.
Never invent a namespace, registry target, base model, or campaign policy.

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
4. **Publish the model.** Follow the consolidation runbook's canonical staging, secret-scan,
   additive-upload, and remote-verification procedure.
5. **Publish traces.** Run `infra/rl_cleanup/make_and_upload_trace_dataset.py` over the complete
   trial set. Keep its hygiene transforms, do not subsample, and verify the dataset before linking it
   from the model card.
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
evidence, storage reclaimed under authority, and all remaining work.
