---
name: rl-standard-launch-iris
description: Launch or relaunch non-agentic MarinSkyRL training on an Iris GPU cluster. Use for standard RL configurations backed by datasets and rewards, without Harbor, Daytona, or terminal-bench agent trajectories.
---

# Standard RL on Iris

Use this workflow for non-agentic RL. It shares the Iris launcher with agentic training but must not introduce Harbor, Daytona, ingress, or literal-agent dependencies.

## 1. Classify and inspect the run

- Work from a clean local checkout and use the current `cloud.iris.launch_rl_iris` interface.
- Inspect the selected YAML and data inputs. Confirm that it has a standard data/reward path (for example, the configured parquet dataset) and does not define a terminal-bench or Harbor agent environment.
- Resolve the intended model, train and validation data, resource topology, checkpoint policy, and artifact destination from the experiment specification and live launcher/configuration behavior.
- Keep database registration disabled unless explicitly authorized.

## 2. Dry-run the actual launch

Provide the required RL configuration and model path to the launcher; add only intentional inputs such as data and node count.

- Run the complete command with `--dry-run` first.
- Verify the generated job identity, image, resource layout, trainer command, checkpoint/artifact destinations, retry policy, and any user-supplied override.
- Prefer the launcher's dynamically resolved defaults for names, rendezvous storage, CPU allocation, image selection, and retries. Override them only when the experiment explicitly requires a different value.
- Do not pass agentic-only arguments or secrets to a standard run.

## 3. Submit and validate progress

Submit the reviewed command without `--dry-run` and preserve the resolved command/configuration with the experiment record.

Verify through the current Iris job interface and durable outputs that:

1. all requested workers are admitted and join the training runtime;
2. data loading and reward computation begin without schema or dependency errors;
3. training advances beyond initialization and emits current metrics;
4. checkpoints appear at the configured destination when due.

A running pod without data, metrics, or advancing steps is not a healthy run. Report the blocked stage and its direct evidence.

## Safety rules

- Fix source or configuration locally, validate it, and relaunch; never hand-edit a live pod or remote checkout.
- Do not cancel a running job without explicit user authorization.
- Preserve logs and checkpoint evidence before an authorized cancellation or cleanup.
