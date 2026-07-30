---
name: rl-standard-launch-iris
description: Validate, submit, and observe standard MarinSkyRL training on Iris for dataset-backed rewards without Harbor, Daytona, terminal-bench, or another agent harness.
---

# Launch standard RL on Iris

Read the selected configuration and current `cloud.iris.launch_rl_iris` interface. Resolve images,
resources, retries, names, capacity, and artifact destinations at execution time.

## Workflow

1. Start from a clean committed revision. Confirm that the resolved configuration uses a standard
   dataset and reward path and contains no agent harness or sandbox environment.
2. Validate model, data, topology, batch geometry, reward function, checkpoint policy, artifact
   destinations, and registration policy.
3. Run the complete launch with `--dry-run`. Inspect job identity, image digest, resources, trainer
   command, input overrides, retry policy, and durable destinations.
4. Submit the reviewed command and preserve its resolved configuration with the experiment record.
5. Verify worker admission, distributed initialization, data and reward processing, advancing
   training metrics, and checkpoints when due.

## Safety

- Do not pass agentic-only arguments or secrets to a standard run.
- Prefer launcher-derived defaults unless the experiment requires an override.
- Never patch a live pod or remote checkout.
- Do not cancel a running job without authority; preserve evidence before an authorized stop.
- A running controller state without advancing batches or steps is not proof of health.
