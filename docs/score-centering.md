# Optional PPO/TIS score centering

Score centering adds a loss term to regular PPO with tokenwise truncated importance sampling (TIS).
SC and its sampled diagnostics are disabled by default. Keep TIS enabled when enabling SC.
The [design](https://echo.oa.dev/wiki/608) describes the correction derived from the
[score-centering paper](https://arxiv.org/html/2609.20807v1).

## Configuration

Use local vLLM and unpacked Megatron with tensor, context and sequence parallel size one.
Data and pipeline parallelism use the existing whole-step loss normalization. The head contains the
sampler's natural top K token IDs and their full-vocabulary probabilities. The trainer scores those same IDs
at the same causal positions, once before the update and again in the training forward.

```yaml
trainer:
  strategy: megatron
  use_sample_packing: false
  algorithm:
    policy_loss_type: regular
    off_policy_correction: tis
    score_centering_topk: 128
    score_centering_enabled: true
    score_centering_diagnostic_tokens: 0
  policy:
    sequence_parallel_size: 1
    megatron_config:
      tensor_model_parallel_size: 1
      context_parallel_size: 1
generator:
  backend: vllm
  run_engines_locally: true
  sampling_params:
    logprobs: 128
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    min_p: 0.0
```

The `tis` preset uses an upper cap of two. A custom correction must contain exactly one token truncation
rule. Distillation, auxiliary FTPO, other policy losses, remote inference and other trainer backends are
rejected. Sampling must preserve the full temperature-scaled distribution; nucleus sampling and other
probability-changing filters cannot supply these behavior probabilities.

Set `score_centering_enabled: false` with positive `score_centering_topk` to measure a matched control.
It retains head capture and old/current rescoring while skipping the correction kernel. Ordinary TIS uses
`score_centering_topk: 0`, `score_centering_enabled: false` and `generator.sampling_params.logprobs: 0`.
It collects only the chosen token's behavior probability and performs no head rescoring.

The sampled action's TIS weight applies to the PPO row. The additive SC row already integrates TIS and is
reduced separately with the same whole-step counts. Multiplying it by that action's TIS weight again changes
the gradient. Outside the captured head, old and behavior probabilities are modeled as proportional to the
current policy, preserving each distribution's measured omitted mass. Small omitted mass alone does not
bound parameter-gradient error or establish a learning benefit.

## Diagnostics

Ordinary `trainer.training_metrics` uses already available chosen-token probabilities and policy-step tags.
`policy/mismatch/*` compares the trainer's pre-update probabilities with the actual generating policy.
Its staleness-zero bucket measures residual trainer–inference mismatch at matching weights. Stale buckets
also contain weight drift. The worker's `policy/log_ratio_*` and PPO clipping metrics compare current
probabilities with the stored old trainer probabilities. An exactly unit PPO ratio can coexist with TIM.

`async/staleness_tokens_*` reports consumed token age in policy steps. Local vLLM stamps tokens at sampling;
tokens without a stamp use the rollout lease's step. These two provenance cases remain distinguishable in
the existing metrics. A lease alone does not prove which weights generated a token.

With an explicitly configured head, set `score_centering_diagnostic_tokens` to a positive integer to sample
at most that many valid tokens per microbatch. `score_centering/sampled_*` reports current/old/behavior
retained and omitted mass, the current-versus-behavior log ratio, and signed/absolute SC loss values.
The existing aggregation averages these microbatch samples across ranks; they are not pooled token means.
Empty microbatches contribute zero, so aggregated retained and omitted mass need not sum to one.
The sampled-token count uses that same averaging.
The sample takes the first valid tokens and does not estimate the response tail. These scalar correction
values measure a loss term, not parameter-gradient change. They require no additional model forward.
Leaving this count zero avoids the new diagnostics. Ordinary telemetry does not enable head capture.

## Snowball qualification and cost

The [qualification record](https://github.com/marin-community/marin/tree/2f92446724b2207222f6f654c52ede00d9e48ac3/experiments/post_training/score_centering_readiness)
contains configurations, every timing window, package versions, replay evidence and allocation accounting.
The executed diagnostic runtime was `fe6c3a9300129af2126e8f81422f728f4d6eda52`, containing product candidate
`2fb64c1c2b5e8692f996942704383be68a242587` and scaffolding excluded from this PR.
It uses Torch 2.13.0+cu132, vLLM `39e62869693c`, Megatron Core 0.18.0 and Transformers 5.16.1;
the recorded lock SHA-256 is `ab752fa67acb209e8ae5ca3ee77b66bc59db1cc1fc3df7a16bc43d6a37c3d86e`.

Snowball 67B/A2B SFT ran on four 8-H100 learner nodes and one 8-H100 inference node in `cw-us-east-02a`.
Learner geometry was PP2/EP8/DP16 with TP/CP/SP1. There were 128 responses from 32 prompt groups,
prompt lengths 108–453 and response lengths 64/256/1024/4096, including padding, mask holes and an empty-loss row.
Independent checks at 381 actual token positions per temperature (1.0 and 0.7) had chosen/head normalizer
errors below 2e-6. Correct-prefix vLLM replay agreed substantially better than a one-token-shift control.
At matching weights, chosen-token absolute TIM had mean 0.03853, p99 0.35906 and maximum 3.37347;
head-entry absolute TIM had mean 0.16161 and maximum 11.18716. These are measured engine differences.

Sixteen controlled optimizer updates and a sampled diagnostic update preceded a native task/admission/GRPO
update. The native step admitted 128 trajectories, consumed tokens with maximum age zero, applied one update
and skipped none. Its sampled current/behavior retained mass was 0.995352/0.995688 and mean absolute scalar
SC loss was 0.001632. Its PPO current/old log ratio was zero while sampled current/behavior absolute log ratio
was 0.06311. Neither mass nor scalar loss establishes parameter-gradient accuracy or learning benefit.

The diagnostic writer failed after those checks on an unused UID field. Saved timing windows and durable
native worker logs support the results; the job's terminal state is failed. Reaching that report line proves
the prior nonzero trainer-drift and frozen-sampler replay assertions passed. Their numerical summaries were
not retained. The evidence separates observations from this control-flow inference and lists missing fields.
The writer repair passed a CPU regression and changes no product code.

Each mode had one warmup and three measured windows in rotating order, with 174080 response tokens,
168386 loss tokens and a 128×4549 sequence tensor per window. Prompt, length, mask and shape workloads match;
generated token content varies. Generation and training run serially with frozen sampler weights and
sequential learner updates. This measures a controlled workload rather than production asynchronous throughput.

| TIS mode | Cycle seconds | Response tokens/s | Old forward / update seconds | Learner peak allocated / sampled device GiB |
| --- | ---: | ---: | ---: | ---: |
| No head, existing trainer telemetry off | 61.97 | 2809 | 0.96 / 6.45 | 54.66 / 79.29 |
| No head, existing trainer telemetry on | 61.27 | 2841 | 0.98 / 6.19 | 54.66 / 78.27 |
| K128 capture/rescore, SC off | 74.86 | 2326 | 3.26 / 10.08 | 54.73 / 78.34 |
| K128 capture/rescore, SC on | 74.33 | 2342 | 3.17 / 9.96 | 55.70 / 79.27 |

Full SC adoption costs +21.29% cycle time against ordinary TIS with existing telemetry, with a descriptive
95% paired interval of +15.77% to +27.07%. Against matched capture, the estimate is −0.72% (−5.62% to +4.44%);
this does not establish a speedup or zero-cost correction. Existing trainer telemetry on/off is −1.13%
(−3.35% to +1.14%), so the 1% target is unproven. New sampled diagnostics remain opt-in; no steady-state
overhead claim is made for them. Existing worker PPO/TIS metrics remain on in every mode.

Intervals use paired log ratios across three repetitions with Student-t assumptions. One allocation, persistent
caches and changing learner weights limit the inference; variation between allocations was not measured.
The measured candidate precedes main's [DP-chunk sharing optimization](https://github.com/marin-community/MarinSkyRL/pull/999).
Peak reserved learner memory was 62.38/61.37/61.44/62.50 GiB in table order. Inference peaks were
57.57 GiB allocated, 58.87 GiB reserved and 64.40 GiB sampled device memory in every mode. Device peaks
were sampled every 0.1 seconds and can miss shorter spikes; allocator caches persist across modes.

Initial weight sync completed 541.5 seconds after admission, including 9.16 seconds in that sync call.
Setup was measured once for the shared allocation. Every owned allocation, including failed starts and
setup/idle/teardown, is accounted for: 25.186 controller-runtime H100-hours, with a conservative provider
allocation upper bound of 27.474 H100-hours. All six jobs and 15 attempts are terminal; owned pods are absent.
