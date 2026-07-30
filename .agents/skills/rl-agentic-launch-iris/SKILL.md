---
name: rl-agentic-launch-iris
description: Validate, submit, and observe agentic MarinSkyRL training on Iris when the configuration uses Harbor, Daytona, terminal-bench, or another sandboxed agent harness.
---

# Launch agentic RL on Iris

Read `.agents/ops/coreweave.md`, the selected configuration, and the current
`cloud.iris.launch_rl_iris` interface before constructing a command. Resolve cluster, image,
credentials, capacity, retry policy, and artifact destinations at execution time.

## Workflow

1. Start from a clean committed revision and classify the harness from the resolved configuration.
2. Validate model, topology, rollout concurrency, sandbox provider, verifier, checkpoint policy,
   artifact paths, and campaign registration/publication policy.
3. Run the complete launch with `--dry-run`. Inspect the generated job identity, image digest,
   resource roles, command, secret references, retries, and durable destinations.
4. Confirm external prerequisites using non-secret evidence: provider access, snapshot headroom,
   task data, model access, and target-cluster image pullability.
5. Submit the reviewed command and preserve its resolved configuration with the experiment record.
6. Observe through initialization into completed trials, verifier rewards, advancing training steps,
   and a checkpoint when due. Use `rl-job-health-deep-dive` when state alone cannot prove progress.

## Safety

- Never expose credentials in commands, logs, or artifacts.
- Never patch live pods or remote checkouts.
- Do not cancel or mutate a running job without authority.
- A running controller state without advancing trials or training is not proof of health.
- On failure, preserve the first causal evidence and recommend the smallest reproducible correction.
