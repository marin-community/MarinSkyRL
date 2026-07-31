# TaskTrove DQ sweep configuration rationale

This document explains coupled settings in `cloud/iris/configs/tasktrove_dq_sweep_30b.yaml`. Read
current values from that file. Named variants have their own overrides; do not apply this rationale
to an override they replace.

## Model and harness

- Preserve the model's native tool-call chat template and pair auto tool choice with the parser for
  that model family.
- Keep the harness version and provider options compatible with the per-trial correlation header.
- Disable harness compaction when its summary turn can produce an unsupported tool call.
- Set the trial timeout from observed healthy-duration percentiles, leaving room for verifier work.

## Reward

- Pass-ratio shaping supplies a useful gradient when binary success is sparse.
- Parser autodetection and binary fallback protect against verifier formats that do not expose a
  test count.
- Apply truncation penalties only to trajectories whose stop reason and original reward satisfy the
  configured rule.
- Keep the preflight gate advisory until its acceptance band has been calibrated for the active
  reward definition.

## Topology and batches

- Policy and reference meshes must use compatible model-parallel geometry.
- Leave gradient-accumulation fusion disabled unless the deployed image contains the required APEX
  extensions.
- Choose inference tensor parallelism to divide the model's KV-head geometry cleanly.
- With sample packing, treat `micro_forward_batch_size_per_gpu` as the number of sequences packed
  into one forward; increasing it multiplies activation and logit pressure.

## Concurrency

Keep trial demand, engine decode supply, and per-coordinator admission aligned using the relation in
`rl-diagnostics.md`. On KV pressure, reduce memory utilization or decode slots before changing model
parallelism.

## Context budget

`context_budget` is the public declaration. The configuration derives the engine window, prompt
cap, per-turn generation cap, and maximum turns from it. Do not set derived fields independently.
Reserve one complete response inside the model window and remember that the trajectory-wide response
budget is distinct from the per-turn cap.

## Run limits

Read the current step/epoch ceiling, artifact paths, image requirement, and per-run overrides from
the configuration and launcher. Do not duplicate those mutable values here.
