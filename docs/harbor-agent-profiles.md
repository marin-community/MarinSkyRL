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
for recorded token IDs and model context. Pi, OpenCode 1.18.2, Mini-SWE-Agent 2.1.0,
Claude Code 2.1.284 and Codex 0.118.0 support this capture path. The four CLI agents
require the vLLM backend, their listed versions and `collect_rollout_details: true`.
Set `model_info.max_input_tokens` to the total context limit and
`model_info.max_output_tokens` to the output limit.
Native API adapters preserve sampled completion IDs, served prompt IDs and
logprobs across tool turns. Each subsequent served prompt extends the preceding
served token stream. Set `trainer.algorithm.tito_full: true` to assemble training
trajectories from the complete served token streams. A loss requiring rollout
logprobs also enables this assembly.

The [FineEnvs multi-harness RL method](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)
uses OpenCode, Claude Code, Codex and Mini-SWE-Agent. Configure those four profiles
with their pinned versions to train with that panel. Other agents still require
capture support before they can supply exact behavior-policy evidence.
