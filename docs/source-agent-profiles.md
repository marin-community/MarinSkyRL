# Source-owned agent profiles

A Harbor task package can ship a validated agent recipe with its prompt configuration.
MarinSkyRL resolves this recipe independently for each task. Tasks without a profile retain
the run's agent configuration. Mixed sources do not share one task-specific prompt.

Include the prompt file in the package and declare it in `task.toml`:

```toml
[metadata.agent_profile]
schema_version = 1
config_file = "agent/setup.yaml"
config_sha256 = "REPLACE_WITH_SHA256_OF_SETUP_YAML"

[metadata.agent_profile.agent]
name = "mini-swe-agent"

[metadata.agent_profile.agent.kwargs]
version = "2.1.0"
```

The configuration file must stay inside the task package. Missing files or checksum
mismatches fail before agent startup; Daytona launch preflight also checks them before GPU
submission. The task materializer preserves the prompt file and metadata in Parquet exports.
Source profiles select agents by name and supply `kwargs`; they cannot request host import
paths. Harbor validates the selected agent name when constructing the trial.

Select the agent in `skyrl.terminal_bench_config.installed_agent.agent` in the launch YAML.
Source settings supply defaults for that configuration. Fields supplied by the run override
them, including each key in `kwargs`. If the run selects an agent whose name differs from
the source profile, MarinSkyRL bypasses the source-specific kwargs and emits a warning.
A run's `kwargs.config_file` is a host path; the source checksum and package-containment
checks apply only to the bundled source file. Harbor records the effective agent settings
in `result.json`. The saved attempt also contains the restored task package and its metadata.

For Mini SWE Agent, override `agent.system_template` and `agent.instance_template` in the
bundled YAML. Keep `{{task}}` in the instance template and preserve the agent's tool-call and
submission protocol: each response must include a bash tool call, and submission is a
separate bash call containing `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
The dataset author must write setup-only instructions in both templates; MarinSkyRL does
not remove repair instructions automatically.
