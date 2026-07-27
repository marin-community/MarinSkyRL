---
name: rl-agentic-launch-iris
description: Launch or relaunch agentic MarinSkyRL training on an Iris GPU cluster. Use when an RL configuration uses Harbor, Daytona, or terminal-bench and must be validated, submitted, and observed through cloud.iris.launch_rl_iris.
---

# Agentic RL on Iris

Use this workflow for RL configurations that execute agent trajectories. It requires only this checkout, its companion Marin checkout, and the Iris credentials/configuration selected by Marin.

## 1. Establish current facts

- Work from a clean local checkout. Make source and configuration changes locally, commit them, and launch from that revision.
- Read the current launcher help and the selected YAML before forming a command. Treat the launcher, the selected configuration, and Marin's current Iris configuration as authoritative for supported flags and resolved defaults.
- Confirm that the selected configuration is agentic: it must define the terminal-bench/Harbor environment expected by the run. Do not use this workflow for a parquet-only standard RL configuration.
- Confirm the requested model, data inputs, resource topology, checkpoint destination, and any experiment-owned artifact destination. Keep database registration disabled unless the user explicitly authorizes it.

## 2. Let the launcher resolve operational defaults

Invoke `python -m cloud.iris.launch_rl_iris` with the required RL configuration and model path, plus only intentional experiment overrides such as data, node count, or selected config overrides.

- Start with `--dry-run` and inspect the resolved command, job identity, resource layout, artifact locations, ingress mode, retry policy, and image reference.
- Accept launcher-generated job names, rendezvous locations, artifact prefixes, CPU allocation, image selection, retry policy, and literal-recording behavior unless the experiment has a documented reason to override one.
- Do not supply an arbitrary source revision, image, Harbor reference, secret value, or hand-built ingress URL. The launcher must derive those from the checked-out code, selected config, and Marin/Iris integration.
- Let the launcher select literal recording from the configured agent harness. Do not force an agentic compatibility flag merely because a prior run used it.

## 3. Submit deliberately

After the dry run is correct, submit the identical command without `--dry-run`. Use the launcher's detach option only when the caller needs a non-blocking submission.

- Preserve the exact resolved launch command and the selected config with the experiment record.
- Do not edit a live pod, cluster checkout, or remote config to repair a launch. Repair local source/configuration, validate it, then submit a new attempt.
- Do not cancel a running job without explicit user authorization.

## 4. Prove the job is productive

Use the current Iris job interface and the durable artifacts named by the launch to verify, in order:

1. the controller has admitted the job and all requested workers join;
2. the serving and training processes initialize without a configuration or credential failure;
3. the agent runner reaches real trials and records rewards or other expected agent outcomes;
4. the trainer emits a later step, loss, and checkpoint/metric evidence.

Do not call a job healthy merely because its pods are running. Report a blocked or failed stage together with the first relevant error and the evidence location.

## Safety rules

- Keep secrets in the approved Marin/Iris secret path; never copy them into YAML, shell history, logs, or artifacts.
- Preserve trial and checkpoint evidence before any authorized cancellation or cleanup.
- If an agentic failure could be a framework defect, reproduce it with the smallest config that still exercises the same launcher path before broad relaunches.
