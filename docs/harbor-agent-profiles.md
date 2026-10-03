# Harbor agent profiles

Set `harbor.agent_profiles` to an ordered list of agent configurations. Each entry
requires a distinct `name` and can override agent fields in `harbor`. Environment,
verifier, model and reward settings remain shared.

```yaml
harbor:
  collect_rollout_details: true
  agent_profiles:
    - name: pi
      version: "0.87.0"
    - name: opencode
      version: "1.18.2"
```

The Harbor dataset assigns a stable index to each task. Directory tasks are sorted
by path; packed tasks use the order of their immutable manifest. A task selects
`agent_profiles[task_index % len(agent_profiles)]`. Every repeated rollout of that
task keeps the same agent, including after batch shuffling or a restart. Changing
the dataset ordering or profile list changes the assignment.

Preflight checks every configured profile against the training loss's requirements
for recorded token IDs and model context. Pi, OpenCode 1.18.2 and Mini-SWE-Agent
2.1.0 support this capture path. Mini-SWE requires the vLLM backend and
`collect_rollout_details: true`; Harbor also requires explicit total-context and
output limits in `model_info`. Its native CLI receives a trial correlation header,
and each subsequent served prompt extends the preceding served token stream.
Use full TITO or a loss requiring rollout logprobs to retain this exact context
in training trajectories.

The [FineEnvs multi-harness RL method](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)
uses OpenCode, Claude Code, Codex and Mini-SWE-Agent. That panel can be assigned
with this interface, but training with that panel remains blocked until the
Claude Code and Codex capture adapters are implemented. Profile selection does not add capture
support to an agent.
