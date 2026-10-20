# Optional PPO/TIS score centering

Score centering adds a loss term to regular PPO with tokenwise truncated importance sampling (TIS).
SC is disabled by default. Occasional bounded probes are enabled in the supported configuration, so ordinary
TIS can report a hypothetical correction before opting into SC. Keep TIS enabled when enabling SC.
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
Between periodic probes it collects only the chosen token's behavior probability and performs no head rescoring.
Set `trainer.algorithm.score_centering_probe.interval: 0` for the probes-disabled timing control.

The sampled action's TIS weight applies to the PPO row. The additive SC row already integrates TIS and is
reduced separately with the same whole-step counts. Multiplying it by that action's TIS weight again changes
the gradient. Outside the captured head, old and behavior probabilities are modeled as proportional to the
current policy, preserving each distribution's measured omitted mass. Small omitted mass alone does not
bound parameter-gradient error or establish a learning benefit.

## Native head transport

Native positive-head requests collect numeric evidence through the pinned vLLM engine's public `generate`
interface with `flat_logprobs`. The existing chat service still renders prompts, detokenizes responses and
parses messages and tools. A request-scoped adapter packs owned NumPy arrays before the response crosses
Ray to the task worker. The internal metadata stays outside the HTTP JSON body. Public chat responses retain
their existing schema; ordinary chosen-token-only TIS keeps its existing path.

For K128, collection retains the old 129-candidate request and reproduces the old chat display limit before
selecting the head. This preserves token identity at tied probability boundaries, including when the sampled
token is inside or outside the natural head. Scores use the full vocabulary normalizer and the pinned chat
API's -9999 floor before token-ID tie ordering. The adapter
clones sampling parameters, isolates concurrent captures and closes the public generator on cancellation.
Empty aborts have zero-row arrays; resumed chunks retain their token alignment, heads and source versions.

The numeric evidence follows this path:

```mermaid
flowchart LR
    A[Public vLLM flat logprobs] --> B[Owned numeric arrays]
    B --> C[Native Ray task]
    C --> D[Existing rollout storage]
    D --> E[CPU collation]
    E --> F[Learner causal scoring and PPO/TIS/SC]
```

The pinned engine still returns CPU primitive lists. Ray transport, the existing S3 group store, padded learner
tensors and transfer to learner GPUs remain in complete native cycles. The existing decoding and source-version
hooks are unchanged. This integration adds no vLLM fork patch, internal override or dependency upgrade.

## Diagnostics

### Periodic probes with SC off

`trainer.algorithm.score_centering_probe` controls three bounds: `interval` in learner policy steps (10 by default),
`topk` (128), and `tokens_per_response` (32). The first eligible prompt group is probed at startup, then
at most one group is selected after each interval. Other groups keep chosen-token-only capture.
The selected group captures its full response heads in vLLM; the task worker then exports at most 32
evenly spaced valid positions per response. The trainer rescores those positions in its existing old-policy
forward. It computes the hypothetical correction with the finalized training advantages and removes all
probe head tensors before the optimizer. SC-off optimizer inputs and updates are unchanged.
The 32-position bound limits export and rescoring. It does not limit vLLM's collection cost: every generated
position in that group has a head, and one positive-head request can make vLLM compute top K for other requests
in the same decode batch. Cadence estimates therefore use the measured complete probe cycle.

Availability is logged at startup and reported as `score_centering_probe/available`. Probes require regular
PPO, one tokenwise TIS truncation rule, local vLLM and unpacked Megatron with TP/CP/SP1. Distillation, FTPO,
other objectives or backends, and an explicitly configured continuous head make periodic probes unavailable.
Continuous capture can use the sampled loss diagnostics below. A selected group may be rejected by normal
admission or lose all trainable positions; a scheduling interval does not guarantee an observation.
`observed`, `sampled_tokens`, `sampled_responses`, `loss_tokens`, `coverage` and `seconds_since_previous`
describe the evidence actually consumed. Coverage is sampled positions divided by this batch's loss tokens;
the sample is one prompt group, so it is not an unbiased population estimate.

Metrics under `score_centering_probe/` include:

- `hypothetical_loss_mean` and `hypothetical_loss_abs_mean`: the actual would-be scalar correction at the
  pre-update snapshot, using this batch's advantages. `ordinary_loss_abs_mean` supplies a scalar scale.
- `unit_advantage_loss_*` and `zero_advantage_fraction`: a separate unit-advantage correction remains
  interpretable when flat-reward groups give zero actual advantages. It is not the correction applied to training.
- `current_retained_mass_mean`, `behavior_retained_mass_mean` and their omitted-mass counterparts:
  full-vocabulary mass in and outside the captured K128 head. `tail_floor_fraction` identifies the tail-floor regime.
- `old_behavior_log_ratio_*`, `sampled_tis_clipped_fraction`, source-age and source-policy-step bounds:
  chosen-token mismatch and the ages of the generating weights. Fresh and stale counts and means are separate;
  unstamped tokens remain unknown rather than being assigned the lease's version.

At this snapshot current and old trainer probabilities are equal (`current_equals_old: 1`). Fresh-source
disagreement with behavior measures engine mismatch at matching weights. Stale-source disagreement mixes
engine mismatch with weight drift; it does not isolate either. These probes do not measure later PPO
minibatches' current/old divergence. Scalar correction size and retained mass do not measure parameter-gradient
change or predict a learning gain. Repeated correction or mismatch across prompt groups can justify a small
matched TIS-versus-SC learning comparison; the comparison must establish any benefit.

Near unit retained mass, subtracting head mass makes the modeled tail sensitive to floating point rounding.
The recorded tail floor fraction flags that regime. A retained Qwen case differed by 0.002755 between FP32
and FP64 scalar replay of the same quantized heads. This does not measure parameter-gradient error;
interpret small corrections together with mass and tail floor metrics in the
[precision evidence](https://github.com/yonromai/marin/blob/e4d85ef4fa57cfdbbd1f652780d3cdf72bb1f161/experiments/post_training/score_centering_integration/evidence/qwen-final-a1-probe-precision.json).

### Routine metrics and continuous-head diagnostics

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

## Current integration qualification

Numeric transport source `87f13fcd82b4677545816e0789c685d10cf051a0`, executed through benchmark runtime
`92d057c561b0893a3d922e42f221eb8d5b1e68fb`, passed 126 CPU adapter checks and a complete 35-update native
Qwen qualification. All four old/new chat comparisons matched exactly at temperatures 1.0 and 0.7 through
4096 tokens; actual causal head references had maximum error below 2e-6. The
[integration record](https://github.com/yonromai/marin/tree/ee2abae52a384675aae56a88d9c916a6286146f0/experiments/post_training/score_centering_integration)
retains source pins, formatter edge cases, complete reports and allocation accounting.

Complete matched Snowball controls used four eight-H100 learner nodes and one eight-H100 inference node on
RNO, with the same checkpoint, tokenizer, tasks, sampling and 174080-response-token batch in both arms. Each
mode had one profiled warmup and two measured rotating cycles, applying 18 updates per implementation.
Independent full-vocabulary references at temperatures 1.0 and 0.7 had maximum error below 1.91e-6.

| Mode | Prior chat objects, seconds | Numeric arrays, seconds |
| --- | ---: | ---: |
| Ordinary TIS | 57.766 | 56.970 |
| Capture/rescore, SC off | 88.132 | 62.982 |
| TIS + SC | 88.381 | 62.248 |
| Probes armed, no observation | 57.683 | 56.993 |
| Complete probe | 63.356 | 57.706 |

SC cycles were 29.57% shorter, with a descriptive log-time interval of 19.51% to 38.38% shorter. Adoption
overhead relative to each allocation's ordinary TIS fell from 53.00% to 9.27%; the new paired descriptive
interval is 7.40% to 11.17%. Two measured cycles and one allocation per arm do not measure population or
allocation-to-allocation variance. Failed starts and the interrupted East baseline are excluded from these
[matched results](https://github.com/yonromai/marin/blob/ee2abae52a384675aae56a88d9c916a6286146f0/experiments/post_training/score_centering_integration/evidence/snowball-controlled-cost-matched-rno.json).

The exclusive SC batch-admission wait fell from 71.69 to 45.42 seconds, including generation and rollout
handoff. The remaining array cycle spent 1.12 seconds assembling the batch, 1.91 preparing old-policy scores,
7.65 training and 6.13 publishing weights. Concurrent frontend profiles and nested timers overlap and must
not be added to these walls. Both optimizer batches retain three 256 MiB head matrices; no GPU memory
reduction is established. CPU flat-logprob lists, array packing, Ray, S3 storage, collation and transfers to
learner GPUs remain.

The interval-ten controlled probe model is 0.166% overhead, with 570.64 seconds between observations. Its
paired descriptive interval is wide, -15.84% to 19.23%, and does not establish a sub-1% ceiling. A full probe
captured 16384 engine tokens and exported 96 valid sampled positions after masks. Defaults remain interval
ten, K128 and 32 positions per response. Real-reward asynchronous throughput and actual observation spacing
remain pending. These serialized cycles do not establish learning benefit.


Fresh asynchronous arms share the [scalar-grader repair](https://github.com/marin-community/marin/commit/5ef1e703d3bae94d510d8cf3f50d902958e10d20),
which rejects unsupported nonscalar answers before SymPy finiteness validation. It passes 437 affected grader
tests and 477 product CPU checks, with three existing dependency skips. A replay of 192 fixed reference-like
replies preserves every old/new grade, including one existing inequality-format failure. No source row is
edited or filtered. The repair changes no SC code, package version, other shared package tree or vLLM pin.
Both fresh arms use the same repaired dependency lock; their timings remain separate from the controls
above, which used the prior grader revision. The
[qualification record](https://github.com/yonromai/marin/tree/03d5715226a493859b8995de411f0ef49da25029/experiments/post_training/score_centering_integration)
retains the parity proof and all incomplete allocation outcomes.

## Prior Snowball qualification and cost

The [qualification record](https://github.com/yonromai/marin/tree/af470a613fb0cc057c748d25ec681dd654821edb/experiments/post_training/score_centering_performance)
contains submitted configurations, complete reports, profiling, independent replays and allocation accounting.
The qualified product is `f6b6ccb1d763486af190da0ff7005ed169285cdc`, executed through diagnostic runtime
`021666fc1b04d9e425321a6279ca19849b4da90f`; the control product/runtime are `325f819f` / `7aad2b37`.
Benchmark scaffolding stays outside this PR. Both include main `32edb00f` and its DP transport changes, with
frozen lock SHA-256 `a9686c1263fae6e77f560c22ab8faa99b694747ac57e09f816ae7ce923c19f99`.

Snowball 67B/A2B SFT ran on four 8-H100 learner nodes and one 8-H100 inference node in `cw-us-east-02a`.
Learner geometry is PP2/EP8/DP16 and TP/CP/SP1; inference uses DP8/EP8. The control completed 21 applied
updates and the candidate 39. Each mode has one profiled warmup and five measured windows in rotating order,
with 128 responses from 32 groups, lengths 64/256/1024/4096, padding, mask holes and an empty-loss row.
Each window contains 174080 response tokens and 164484 loss tokens in a 128×4549 sequence tensor.
Cycles include native task/admission/storage, old scoring, optimizer work, normal metrics and verified sync.

| TIS mode | Control cycle s | Candidate cycle s | Candidate response tokens/s | Learner peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Ordinary, probes off | 58.553 | 58.237 | 2989 | 54.558 |
| K128 capture/rescore, SC off | 111.818 | 84.434 | 2062 | 54.611 |
| K128 capture/rescore, SC on | 109.780 | 84.734 | 2054 | 55.535 |
| Probes armed, no observation due | — | 58.033 | 3000 | 54.560 |
| Complete probe cycle | — | 63.645 | 2735 | 54.559 |

SC cycles fell 22.81% between allocations (descriptive Welch 95% interval 21.06–24.52% reduction).
Full adoption still costs 45.50% against ordinary TIS (paired 95% interval 43.38–47.63%), compared with
87.49% in the same-base control. Capture alone costs 44.98%; SC versus matched capture is 0.36%
(−0.29–1.01%). This does not establish zero correction cost. The remaining cost is mainly native head
collection and chat response construction/transport, followed by continuous head scoring.

The implementation skips head-text decoding and unused display bytes for private exact-token captures,
preserves raw numeric arrays and safe int32 IDs, and reduces parsing/collation work. Response detokenization,
public responses, probabilities, source stamps and the full-vocabulary normalizer are preserved. Native
profiles confirm head-token decoding is removed; alternative object constructors showed no useful benefit.
The native array transport above is a follow-up to this qualification and has its own source pins and evidence.
K32 remains an optional approximation without a Snowball learning or parameter-gradient justification here.

A complete long-response probe adds 5.612 s over the armed path (95% mean interval 4.493–6.732 s), including
all four 4096-token response heads and 96 valid sampled positions after masking. The default interval of ten
models 0.61% average overhead versus off (−0.17–1.40%), about 0.96% incremental cost over the armed path.
It does not prove overhead below 1%. Observations would be 585.94 s apart in this serialized workload,
starting immediately. Per-probe coverage is 0.0584% of loss tokens, or about 0.00584% across ten equal cycles.
Ordinary and armed optimizer tensor byte counts match the probe cycle. Due/non-due paths were measured
separately; the post-qualification default change from 100 to 10 changes cadence only.

Six further updates used normal rewards, masks and finalized GRPO advantages at interval two. Three probes
each observed 128 fresh-source positions, with no stale or unknown tags. Actual absolute scalar corrections
were 0, 0.000460 and 0; the unit-advantage values were 0.001209, 0.000798 and 0.001678. The two zeros came
from zero advantages. Current retained mass ranged 0.97484–0.99285, while fresh chosen-token mismatch means
were 0.0417–0.0612 and PPO current/old log ratios stayed zero. These are distinct signals, not learning gains.
Independent NumPy FP64 replay matches the nonzero actual correction to 1.12e-7; unit-advantage differences
reach 1.98e-6 near the FP32 tail floor. Counts, coverage, zero-advantage and tail-floor metrics limit interpretation.

The pinned Qwen startup check passed exact native chat tokens, message, source-version and chosen/head-score
A/B checks at temperatures 1.0 and 0.7, with actual decoder suppression. Snowball checked 381 actual causal
positions per temperature against a full-vocabulary reference, with maximum head error below 2e-6, and checked
compact native positions at 0.7. All native updates applied once, skipped none, and consumed stamped source
age zero. The earlier diagnostic native 0.7 warmup had an engine override error and is excluded; the final
candidate records effective engine temperatures. Its control's measured 1.0 windows are unchanged.

Inference peaks reached 58.309 GiB allocated, 58.877 GiB reserved and 64.406 GiB sampled device memory.
Memory sampling every 0.1 seconds and persistent allocator caches limit incremental comparisons. Admission to
initial sync took 638.4/575.0 s for control/candidate, including 9.621/9.171 s in the sync calls. This follow-up
accounts for 98.360 H100-hours from recorded admission or earlier Qwen controller-start proxies, with a
110.275 H100-hour conservative provider upper bound. Failed starts, setup, idle time and teardown are included.
All eight follow-up jobs are terminal with zero retries; owned pods are absent on both H100 clusters.

These are serialized native cycles, not production asynchronous throughput. Five repetitions and one allocation
per implementation give descriptive intervals; changing weights, caches and placement variation limit inference.
Admission rejection, response lengths and mixed decode batches can change average probe cost and cadence.
The [older readiness benchmark](https://github.com/marin-community/marin/tree/2f92446724b2207222f6f654c52ede00d9e48ac3/experiments/post_training/score_centering_readiness)
used a different raw/cohort loop before main's transport changes; its 21.3% result is historical context.
No parameter-gradient accuracy or learning benefit is established.
