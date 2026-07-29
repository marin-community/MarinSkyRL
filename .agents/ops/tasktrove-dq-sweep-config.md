# `tasktrove_dq_sweep_30b.yaml` — settings that are not self-explanatory

Facts only. The config carries no comments; this file owns the reasons. Update here when a
setting changes.

## Model and harness

- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`. Its native HF chat template emits the Qwen3-Coder
  XML function-call format. Do not set a custom chat template — a mismatched one drives
  `tool_use` to zero.
- `enable_auto_tool_choice: true` + `tool_call_parser: qwen3_coder` are required. opencode sends
  `tool_choice: "auto"`; without them the engine errors, opencode exits 1, and every rollout
  scores zero.
- Harness is opencode, pinned to `1.18.2`. Earlier 1.0.x drops provider options and loses the
  per-trial correlation header the literal bridge needs.
- `opencode_config: {}` keeps auto-compaction OFF. Compaction's summary turn can emit a tool
  call, which opencode rejects fatally, ending the trial at reward 0.
- `override_timeout_sec: 1800` sits above the healthy-trial p90 (~19 min). A lower ceiling kills
  healthy trials mid-task.

## Reward definition

- `enable_reward_shaping: true` + `reward_shaper: pass_ratio` score the fraction of tests passed
  rather than overall pass/fail. Binary scoring puts nearly every trial at 0.0, which is the
  sparse regime RLOO cannot learn from.
- `reward_parser: null` auto-detects the test framework from verifier stdout.
- `reward_shaping_fallback: true` falls back to the binary reward when stdout will not parse.
- `truncation_penalty: 0.25` is subtracted when a trajectory has `stop_reason == "length"` and
  `original_reward == 0`. Untuned first value.
- `preflight_gate` runs in `warn` mode. Its 0.25–0.75 band was calibrated on binary rewards and
  has not been recalibrated for `pass_ratio`, so it must not abort runs yet.
- Requires an image at or after `gpu-rl-megatron-ac5a9c65`. See
  [gpu-rl-image-build.md](gpu-rl-image-build.md) for which paths are baked.

## Meshes

- Trainer: `strategy: megatron`, TP4 × PP2 × CP1 = 8 model-parallel, DP=2, EP4 over 128 experts,
  16 GPUs. Policy and ref use the same geometry.
- `gradient_accumulation_fusion: false` on both policy and ref: APEX in the image is not built
  with `--cpp_ext --cuda_ext`, so `True` crashes at `init_model` with
  `fused_weight_gradient_mlp_cuda is not found`. Gradients are identical unfused. Removing this
  requires rebuilding the image with those APEX flags.
- Inference: 4 engines × TP4, EP4, DCP1, 16 GPUs. TP4 splits the model's 4 KV heads exactly one
  per GPU, so no KV replication and no DCP is needed.
- Total footprint 4 nodes × 8 H100 = 32 GPUs, with `colocate_all: false`.
- `micro_forward_batch_size_per_gpu: 1`. The log-prob forward chunks the mini-batch into groups of
  this many sequences (`TensorBatch.chunk` takes a chunk *size*, not a chunk count), and with
  `use_sample_packing: true` a group of N is packed into one N× longer forward. At a 131,072-token
  window that multiplies the activation and vocab-parallel logit peak by N. Policy ranks already
  sit near the card limit, so anything above 1 leaves no headroom for the training backward.

## Concurrency

- `max_num_seqs: 24` × 4 engines = 96 decode slots. This is the supply lever.
- `n_concurrent_trials: 96` is the demand lever and must be moved in lockstep with it. Never
  permute one side alone.
- The rollout dispatcher fans out over `num_coordinators: 4` actors and divides
  `n_concurrent_trials` by that count, so each coordinator admits 24 trials and the cluster-wide
  ceiling stays at 96.
- On a KV-bound OOM, lower `gpu_memory_utilization` (0.80) or `max_num_seqs` first. Never TP.

## Context budget

`context_budget` is the ONLY public declaration. Every length field below is derived from it, and
the config is forbidden from setting them directly (`_validate_no_derived_context_fields` raises).

| declared | value |
|---|---|
| `request_window_tokens` | 131,072 |
| `max_new_tokens_per_turn` | 16,384 |
| `max_turns` | 90 |
| derived `max_input_tokens` | **114,688** (= window − per-turn) |

Materialized into: `engine_init_kwargs.max_model_len` 131,072; `generator.max_input_length` and
`trainer.max_prompt_length` 114,688; `sampling_params.max_generate_length` 16,384;
`generator.max_turns` and `harbor.max_turns` 90; `model_info.max_input_tokens` / `max_output_tokens`
114,688 / 16,384.

Prompt cap plus one full response equals `max_model_len` exactly, by construction — the derivation
reserves one complete response and no more, so there is no slack.

A second, DIFFERENT cap lives in the generator: `max_response_tokens = max_generate_length +
max_input_length − initial_prompt_length`, applied to the whole trajectory. `stop_reason ==
"length"` — the condition `truncation_penalty` fires on — is set against THAT, not against the
per-turn cap. The per-turn cap severs tool-call JSON without setting it.

Raised from 32,768 / 4,096 / 30 on 2026-07-27. At 4,096 per turn, 28% of turns on one arm ended
exactly at the cap and 6.1% of all tool calls were rejected as malformed JSON — the model writes a
whole file in one call and the cap severs it, 3× more often on zero-reward trials.

## Budget

- `max_steps: 80` or 2 epochs, whichever comes first.
- `trials_dir` and `ckpt_path` are overridden per-run by the launcher.
