# Tune vLLM for Snowball and Hero

[Choose the task first](performance.md). This page covers BF16 standalone serving and MarinSkyRL rollouts on H100 and GB200/B200. Pick a measured layout closest to your lengths and objective, then check queues, KV pressure and useful output. Numbers below are historical measurements, checked on 30 September 2026.

**Preserve the RL numerical contract when tuning a rollout server.** Hero's 4K qualification uses non-eager, batch-invariant serving and captured expert routes for learner replay. Serving batch invariance OFF, eager mode, and a different attention backend lack a valid same-token online cost/safety pair. Longer contexts are performance-only probes. [Qualification](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5835032707); [numerical tradeoffs](performance-megatron.md#performance-and-numerical-alignment).

## Standalone serving

Use the installed Marin wheel and a locally staged, identified BF16 export. Resolve the current [Marin GPU artifact](https://github.com/marin-community/marin/blob/main/config/external/vllm/gpu.toml), then inspect the installed version and supported arguments. The measured trials used vLLM source `25f0fc1a` over precompiled kernels; a newer wheel is a new validation condition.

**Hero release gap:** on 30 September the promoted Marin artifact and draft [SkyRL runtime #799](https://github.com/marin-community/MarinSkyRL/pull/799) still pinned `01911be34fac`, which predates the qualified split-expert loading and bounded reload. Hero recipes below require the identified trial assembly or a subsequently qualified immutable wheel. A frozen install of those old pins does not reproduce Hero qualification.

The measured short-context serving geometry was TP1/PP1 with data and expert parallelism over the serving fleet. Snowball H100's eight-GPU example fits one eight-GPU node. Hero and the 16-GB200 examples require the tested cross-node placement and network. Stage a checkpoint once per node; verify the actual nodes and memory before model load. CoreWeave storage policy is in [the Iris operations guide](../.agents/ops/coreweave.md).

For a latency objective, reuse the closest measured layout but begin with your expected live concurrency, below its throughput point. Measure p95 first-token and whole-response latency at the expected burst. None of the capacity rows establishes an interactive latency service limit. For bulk throughput, retain replacement clients during the measured window, and also count whole-bank completion including the tail.

<details>
<summary>Snowball one-node server argument sketch</summary>

With an eight-H100 allocation, the supported installed fork, and a staged export, these are the relevant arguments to resolve into the normal deployment. This sketch does not submit a job or specify credentials/networking:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve /path/to/staged-snowball \
  --dtype bfloat16 --tensor-parallel-size 1 --pipeline-parallel-size 1 \
  --data-parallel-size 8 --enable-expert-parallel \
  --max-model-len 4096 --max-num-seqs 1024 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching
```

The RL producer also retained routes and processed token log-probabilities. Those options and the weight-transfer lifecycle belong to the SkyRL configuration. Match the intended sampling/stop policy; forced-length capacity outputs do not measure answer quality.

</details>

## Measured serving examples

These BF16 non-eager, batch-invariant runs used a forced-length GSM8K bank capped at 4K. All kept prefix caching off. `max_num_seqs` is per engine; client concurrency is over the fleet. Rates are aggregate **supplied decode**, not whole-bank or RL rates.

| Model / hardware | Serving GPUs; engine sequence / batched-token caps | Tested clients → decode tok/s | What the row supports |
| --- | --- | --- | --- |
| Snowball / H100, 26 Sept | 8; 1,024 / 4,096 | 4,096 → 32,967, zero KV preemptions | Short bulk/rollout capacity reference; 8,192 clients overload KV. [H100 evidence](#serving-evidence) |
| Hero / H100, 28 Sept | 32; 64 / 512 | 768 → 3,558, zero preemptions; 1,280 → 5,243, 120 preemptions | 768 is the lower-pressure measured point. At 1,536, 5,690 tok/s costs 448 preemptions for only 1.7% more whole-bank rate than 1,280. [Evidence](#serving-evidence) |
| Snowball / GB200, 28 Sept | 16; 512 / 4,096 | 4,096 → 29,354, zero preemptions | Curve still rising. Smaller fleet and interactive limit unknown. [Evidence](#serving-evidence) |
| Hero / GB200, 29 Sept | 16; 256 / 512 | 3,072 → 9,142, zero preemptions, peak sampled KV 88.4% | At 4,096 clients: 9,960 tok/s, 100% KV, 218 preemptions. Eight-GPU isolated sizing remains unknown. [Evidence](#serving-evidence) |

Do not turn these points into a model-speed factor: server caps differ, and online requests are supplied differently. The window covers 20–80% of completions while at least one full client wave remains. A reported zero preemption count means a fresh complete window, not a missing series.

## SkyRL rollout serving

Use the current [standard](../.agents/skills/rl-standard-launch-iris/SKILL.md) or [agentic](../.agents/skills/rl-agentic-launch-iris/SKILL.md) Iris launch path and inspect its resolved configuration. `generator.max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `enable_prefix_caching`, `enforce_eager` and `engine_init_kwargs` describe the server; rollout admission controls its supply. The config schema is [ppo_base_config.yaml](../skyrl-train/skyrl_train/config/ppo_base_config.yaml).

Use cap64/512 for the tested Hero H100 online path and cap16/512 for the tested Hero GB200 online path. Raising H100 cap64→128 reduced accepted RL throughput 8.7% with 90 versus zero KV preemptions. GB200 cap16→32 gave 89.56→88.39 accepted response tok/s with zero preemptions in both arms. These four-step samples have independently sampled outputs; they do not isolate causal cap cost. [Exact online runs](performance-megatron.md#online-evidence).

Snowball's [worked 32+8 H100 example](performance-megatron.md#worked-task-short-snowball-math-rl-on-h100) used cap1,024/4,096. Its warm server queues and learner wait were low. A larger sequence cap or serving fleet has no demonstrated value for that window.

## Read the signals together

For SkyRL, open [RL Post-training (async)](https://grafana.oa.dev/d/marin-async-rl), select cluster and exact run, and verify fresh engine-labelled samples. [Grafana run selection](grafana-rl-runs.md) and the [metrics contract](design/vllm-metrics-contract.md) explain the service filter: SkyRL exports `service=marinskyrl`, `metric_source=vllm`; standalone `/metrics` uses `service=vllm`. The [learner guide](performance-megatron.md#telemetry-you-can-use) gives FineLog access and bounded SQL.

| Signal and source | Unit / window | Supports the hypothesis | Contrary evidence and next action |
| --- | --- | --- | --- |
| Native `generation_tokens_total`, `prompt_tokens_total`; `/metrics` or FineLog | Counter delta / elapsed seconds, per physical engine then fleet, 5 min or exact ordinary phase | Decode rises with more supply while latency stays acceptable | Flat decode with a growing prefill load: try the batched-token budget; low running counts: check upstream supply |
| `num_requests_running`, `num_requests_waiting` | Requests per engine, median and maximum over the same window | High waiting with busy engines suggests admission/server pressure | Idle engines, low queue and low KV during learner wait: inspect groups, tools, admission or pauses |
| `kv_cache_usage_perc`, `num_preemptions_total` | Native KV fraction 0–1; reset-aware preemption delta per window | High KV plus preemptions supports cache pressure | Low KV and zero preemptions weaken it; inspect CPU dispatch, kernels and communication instead |
| `time_to_first_token_seconds`, `request_queue_time_seconds`, `e2e_request_latency_seconds` | Histogram interval p95 and count, seconds, 5 min at expected burst | Lower queue/TTFT meets a latency goal | Decode gain with p95 above the limit is a failed latency tradeoff |
| `request_time_per_output_token_seconds` | Histogram interval, seconds per output token per completed request | Better sustained response speed | It is a request average, not a native inter-token histogram or aggregate fleet capacity |
| `request_success_total` by finish reason; bridge `request_outcome_count` | Reset-aware native completions and bridge outcomes per window | Complete useful answers without length/error growth | More truncated, abort/error or timed-out output reduces useful work; inspect the retained trajectory and first error |

Missing, stale or dropped telemetry leaves the hypothesis **unknown**. Current SkyRL `vllm/peak_gpu_cache_usage_perc` summaries are percentages, unlike native fractional KV. Do not average p95s across engines or difference a gauge. Histogram interval quantiles require bucket deltas per complete series; preserve engine identity and reset handling. [Typed source](../skyrl-train/skyrl_train/inference_engines/vllm/stats.py).

Names in the table are exported series; standalone Prometheus prefixes them with `vllm:`. The in-process snapshot API uses base counter names such as `generation_tokens`, `num_preemptions` and `request_success`, which SkyRL projects to the `_total` series. Inspect the exported schema before selecting a name.

<details>
<summary>One adjustment after identifying the limit</summary>

| Knob | When to try it | Performance and correctness boundary |
| --- | --- | --- |
| Client/admission concurrency | Server idle and useful groups scarce | Increase supply within the current policy-age contract. More workers alone may not expand the admission window |
| `max_num_seqs` | Waiting requests, memory headroom, a server cap reached | Can raise batch throughput and KV pressure together. Check tails/preemptions and whole-RL acceptance |
| `max_num_batched_tokens` / chunked prefill | Prefill displaces decode or TTFT is poor | Lower budget favors decode; larger favors prefill throughput. Test the task's latency and throughput together. [vLLM tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) |
| KV allocation / replicas / EP layout | Confirmed KV pressure | Raising memory utilization trades headroom for cache; another replica costs GPUs. Recheck startup, longest requests and publication peaks |
| Prefix caching | Repeated prompt prefixes | Historical Hero hits were 30–34%, but after-pause rate was 755.78 OFF versus 755.51 ON. The probe paused/resumed without a weight transfer. Keep off for the tested online path; reuse across actual publication is unmeasured. [Evidence](#serving-evidence) |
| CUDA graphs / eager mode | Startup cost dominates a short standalone job | Eager skips graph capture but its steady-state cost is workload-dependent. Hero's non-eager RL path is qualified; eager RL has no same-token pair |
| Attention / MoE backend, batch invariance | A supplied plateau without cache pressure, or numerical mismatch | Keep Hero's qualified FA2/replay/invariance package. A kernel change needs fixed-token numerical checks and online clipping/accepted-rate evidence |
| CPU scheduling / communication | Low KV, supplied workload, flat rate | Inspect event-loop/dispatch spans and rank or host traces. A CPU span includes waits; it does not prove GPU kernel utilization |

</details>

## Context and qualification

Hero's permanent export remains 4K. Disposable probes raised HF, rotary and engine limits together and served actual requests through 256K: 64 H100 with PP2 or 16 GB200 with PP1. These sparse-client banks used one-token outputs and selected 64-token continuations. They do not establish long-response capacity, RL correctness, or multi-turn quality. The 32-H100 PP1 layout had roughly 208K KV slots per engine, so 256K could not fit there. [Probe evidence](#serving-evidence).

Snowball longer-context rates on these H100/GB200 layouts are **unknown** in this guide. For an unfamiliar context, first check actual peak input plus output, longest-tail cases, engine KV reservation and all position bounds. Increasing the configured maximum for unchanged request histories does not itself measure a longer-context workload.

### Serving evidence

<details>
<summary>Exact historical runs, sources and durable reports</summary>

The `s3://` locators require authorized CoreWeave access; they are durable run artifacts, not public HTTP pages. Inspect a report's config and source revisions before reproducing it. Fetch only the small report/reduction needed, not checkpoints or traces. The H100 source record is the [26 September sizing comment](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5848726769).

| Result | Iris job; source revision | Report locator |
| --- | --- | --- |
| Snowball H1008 | `/romain/snowball-capacity-h100-01a0dca4-a8`; `b8c37cc90111795c3cccb355b0de163796280b8f` | `s3://marin-us-east-02a/marin/users/romain/hero-learning-01a0dca4/snowball-capacity-h100-a8/analysis/report-bb47d02f3c5f23179dbcf6b58aa58939f6aeb837e31ba0abe875d40df2f1e761.json`; SHA256 `bb47d02f3c5f23179dbcf6b58aa58939f6aeb837e31ba0abe875d40df2f1e761` (211 MiB raw bank) |
| Hero H10032 knee | `/romain/hero-h100-serve-32g-knee-01a0e406-a1`; `9ffb4919322b4bd67cb51b05c98332f506e644c1` | `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-serve-32g-knee-a1/report.json`; SHA256 `3093ac5959d84e2f2be00fd0e5d657481e184bd53b9b392e8e364f997d159dab` |
| Snowball GB20016 | `/romain/gb200-snowball-serve-16g-01a0e406-a1`; same source as knee | `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/gb200-snowball-serve-16g-a1/report.json`; SHA256 `f9b73d0d02244dbc07ea27aa36b6d7b1a2c7a1c0cc41212da5206fea1df0fe40` |
| Hero GB20016 cap256 | `/romain/hero-gb200-serve-16g-cap256-01a0e406-a1`; `dfdc80be6643f49af8d22ab9adda5319ae34befa` | `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/gb200-serve-16g-cap256-a1/report.json`; SHA256 `04b728f72b5dd8a67b19efb4b4c884d5c525da531685e3bcc2f4cd47012c289a` |
| Hero H100 prefix pair | `/romain/hero-h100-prefix-on-32g-01a0e406-a2` / `/romain/hero-h100-prefix-off-32g-01a0e406-a1`; source `577a5db16b3b971a96ba7254db2e89f60dd83cbb` | `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-prefix-on-32g-a2/report.json` and `h100-prefix-off-32g-a1/report.json` under the same parent |
| Hero GB200256K / H100256K | `/romain/hero-gb200-long-16g-256k-01a0e406-a1` / `/romain/hero-h100-long-64g-256k-pp2-01a0e406-a2` | `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/gb200-long-16g-256k-a1/report.json` / `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-long-64g-256k-pp2-a2/report.json` |

All short supplied curves use vLLM `25f0fc1aae71cccb6da65046cff13145512a6946`, TP1/PP1, memory utilization 0.90 and bank SHA256 `b15aeac7c3eb714fe45fc6b8e274342ad26f00970d84f2b7985b42c422b4e905`. Hero online H100 used 0.92. Current [vLLM Grug source](https://github.com/marin-community/vllm/blob/39e62869693c/vllm/model_executor/models/grugmoe.py) and [NVIDIA parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) explain mechanisms, not these measured rates.

</details>
