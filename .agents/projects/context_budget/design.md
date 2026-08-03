# Context budget normalization

## TL;DR

Iris RL configs will declare one context budget. The launcher derives every
SkyRL, vLLM, and TerminalBench length field from it, records the resolved
values with the run, and rejects direct declarations or overrides of those
derived fields. Step-wise generation will clamp each request against the
actual tokenized prompt before sending it to vLLM.

## Background

`cloud/iris/configs/tasktrove_dq_sweep_30b.yaml` independently declared a
131,072-token vLLM window, a 130k/16k Harbor model budget, and a
999,999-token SkyRL multi-turn input limit. That configuration admitted an
approximately 45k-token retained-history trajectory, then OOMed PPO backward
on H100 at step 7. `cloud/iris/rl_config_translation.py` already owns the
translation from Iris YAML to Hydra arguments, making it the correct boundary
for a single length contract. `skyrl-train/skyrl_train/generators/step_wise_generator.py`
currently checks length only after a request has completed.

## Proposal

Each Iris config declares:

```yaml
context_budget:
  request_window_tokens: 32768
  max_new_tokens_per_turn: 4096
  max_turns: 30
```

`request_window_tokens` is the maximum tokens in a single vLLM request.
`max_new_tokens_per_turn` is reserved from that window. The client input
allowance is `request_window_tokens - max_new_tokens_per_turn`; the launcher
uses it for SkyRL's initial-prompt admission filter, SkyRL multi-turn input
limit, and TerminalBench/LiteLLM `model_info.max_input_tokens`. It uses the
output allowance for SkyRL sampling and
`model_info.max_output_tokens`, and writes the request window to vLLM
`max_model_len`. It passes `max_turns` to both SkyRL and Harbor's current
`max_turns` agent argument; the deprecated Harbor `max_episodes` field is no
longer configurable.

The exact tokenized prompt is available immediately before each step-wise vLLM
call, so no guessed static protocol margin is needed. The runtime guard lowers
that request's output limit to the remaining window and stops before a request
with no room for output. Tool observations and chat-template overhead therefore
cannot make a final request exceed vLLM capacity.

`cloud/iris/training_driver.py` writes `resolved-context-budget.json` beside the durable
Harbor trial bundle when it is an object-store URI, otherwise beside local run
artifacts, and prints its fields before training. Low-level YAML
declarations and `--skyrl_override` values for derived length fields fail at
launch. High-level `context_budget.*` overrides are resolved before Hydra
arguments are built.

## Scope

All `cloud/iris/configs/*.yaml` files migrate in this change. The TaskTrove
OpenCode configurations use 32,768 request tokens, 4,096 output tokens, and 30
turns. Existing non-TaskTrove profiles retain their intended request windows,
output budgets, and turn limits, but gain coherent input budgets.

This does not infer a training-safe window from vLLM capacity. A topology/model
calibration registry is a separate follow-up: vLLM KV-cache fit does not prove
that PPO backward fits activation memory.

## Open questions

The normal retained-history profile has no episode-wide generation cap. If a
future agent compacts history, it may need a separate rollout-cost policy; that
is intentionally outside this context-safety schema.
