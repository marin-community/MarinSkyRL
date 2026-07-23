# Research

- `cloud/iris/rl_config_translation.py` parses each Iris YAML and is the only
  supported source for Hydra arguments in `cloud/iris/run_rl.py`.
- `skyrl-train/skyrl_train/config/ppo_base_config.yaml` defines the downstream
  defaults: `trainer.max_prompt_length`, `generator.max_input_length`,
  `generator.max_turns`, and `generator.sampling_params.max_generate_length`.
- `skyrl-train/skyrl_train/utils/utils.py` derives PPO `max_seq_len` from
  `generator.max_input_length + max_generate_length`; coherent derived values
  preserve its request-window interpretation.
- `skyrl-train/skyrl_train/generators/step_wise_generator.py` emits a request
  before checking the accumulated input length, so a late multi-turn request can
  exceed `engine_init_kwargs.max_model_len`.
- `cloud/iris/configs/tasktrove_dq_sweep_30b.yaml` and its `ncclnet` and
  `terminus2` variants contained the r9 131k/130k/16k/999999 configuration.
- TerminalBench configs expose their agent-facing limits under
  `terminal_bench.model_info`; this maps to the Harbor/LiteLLM model contract.
- `skyrl-train/examples/terminal_bench/harbor_config.py` forwards Harbor agent
  limits. Harbor Terminus 2 treats `max_episodes` as deprecated and gives the
  current `max_turns` argument precedence, so the schema derives `max_turns`
  instead of carrying a second agent-loop limit.
