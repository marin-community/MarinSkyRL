# Tune Megatron and the asynchronous RL loop

[Choose the task](performance.md), then choose a learner and inspect the whole loop. These are BF16 Snowball/Hero measurements on CoreWeave H100 and GB200/B200, checked on 30 September 2026. A learner-only rate excludes rollout supply, scoring, publication and checkpointing.

**Hero learning is qualified at 4K on non-agentic GSM8K with route replay, frozen query bias and MuonH/AdamH/Adam.** The longer synthetic AdamW trainer probes and the short staleness-two supply diagnostic have different qualification boundaries. Read [the learning audit](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5860256429) before changing that contract.

## Choose a learner

The fixed bank has 128 sequences and 116,351 unpadded prompt + response tokens. Warm medians use three later updates. Rates divide those tokens by training time and learner GPUs. Peak HBM is maximum allocated memory over ranks/updates; host memory is peak cgroup GiB per node. None of these is a measured device-utilization or MFU number.

| Model / hardware; measured 26–28 Sept | Learner GPUs; TP / PP / EP / CP | Warm s/update; sequence tok/s/GPU | Peak HBM / host GiB | Supported starting decision |
| --- | --- | --- | --- | --- |
| Snowball H100 | 32; 1 / 2 / 8 / 1 | 5.57; 653 | 50.4 / 120.4 | Efficient tested fixed-bank point; AdamW. [Evidence](#learner-evidence) |
| Snowball GB200 | 16; 1 / 2 / 8 / 1 | 10.45; 696 | 81.6 / 111.0 | Efficient tested point. 32 GPUs takes 6.16 s but 17.9% more GPU-seconds. [Evidence](#learner-evidence) |
| Hero H100 | 96; 1 / 24 / 4 / 1 | 19.97; 60.7 | 75.0 / 449.9 | Fixed-bank efficiency option; **online batch256 OOMed**. Use the measured 128-GPU online layout for the large-batch example. [Evidence](#learner-evidence) |
| Hero GB200 | 48; 1 / 12 / 4 / 4 | 60.36; 40.2 | 128.6 / 445.1 | Efficient tested MuonH point. 64 GPUs at PP16 takes 52.13 s but 15.1% more GPU-seconds. [Evidence](#learner-evidence) |

Choose PP for layer/state fit, EP for experts, CP for sequence memory, and DP for replicated throughput. Resolve actual rank groups from the launch plan: attention DP and expert DP differ, and Hero can fold EP across the CP/DP mesh. Do not multiply every printed dimension to infer physical GPUs. Snowball32 TP1/PP2/CP1 has attention DP16 and expert DP2; Snowball16 has DP8 and expert DP1. The provider owns valid head geometry and Hero's local/global attention choices. [Grug training](grug-megatron-training.md); [Megatron parallelism](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html).

Start with packed samples, micro-forward = micro-train = 1, the tested recomputation and optimizer offload settings, and explicit longest sequence and group batch. For Snowball, [snowball_megatron_full.yaml](../cloud/iris/configs/snowball_megatron_full.yaml) shows the current backend fields; its batch/serving allocation is a different workload from the example below. Match a historical resolved config deliberately, then validate the current launcher.

## Find the full-loop limit

Inspect at least two warm ordinary cycles after initialization. Use a longer window when response tails are variable. Report startup/evaluation/save separately and include them in total task cost. Old `timing/*` spans overlap; current exclusive `timing/step_wall/*` budgets partition the step. Do not sum overlapping timers to manufacture a wall time.

| Observation | First hypothesis | One next change; what would confirm it |
| --- | --- | --- |
| Learner waits; servers have low running counts, low KV, no queue | Group tails, tool waits, admission, scoring or publication starving supply | Inspect group/lease and pause windows. Change the identified admission or worker bound within the allowed age; accepted tokens/GPU-hour must rise without worse quality/drift |
| Learner waits; servers queue and preempt | Serving/cache pressure | Use the [vLLM guide](performance-vllm.md#read-the-signals-together). Try one cap or fleet change; lower wait with higher accepted work confirms it |
| Training dominates; memory fit is tight | Learner layout/activation cost | Try a compatible PP/EP/CP or recomputation option on the same bank. Lower GPU-seconds and safe peak memory confirm it; CPU offload may move cost to host transfer |
| Scoring dominates | Reference/replay or mesh forward cost | Compare same tokens/routes/weights and measure full-loop benefit; changing routes can fail the numerical gate |
| Publication dominates | Pause, optimizer offload, rank alignment, transfer, reload or drain | Split the publication stages. A shorter broadcast timer alone does not identify network bandwidth; require exact installed-weight checks and accepted-rate gain |
| Few groups carry mixed rewards; raw reward high but truncation grows | Useful learning work or termination is failing | Inspect complete answers, verifier and informative-group fraction before buying GPUs. Historical Snowball raw reward hid nontermination. [Evidence](https://github.com/marin-community/marin/issues/8936#issuecomment-5581680479) |

### Telemetry you can use

Open [RL Post-training (async)](https://grafana.oa.dev/d/marin-async-rl) with the exact cluster/run and time window. Use the [run-selection guide](grafana-rl-runs.md) and [metric definitions](design/async-rl-telemetry.md). FineLog commands run from a Marin checkout; authenticate through the normal Iris access path. Discover the current namespace/schema before adapting SQL:

```bash
uv run finelog namespaces cw-us-east-08a
uv run finelog schema cw-us-east-08a telemetry_v1.marinskyrl
uv run finelog query cw-us-east-08a --format table < bounded-query.sql
```

Use `marin` for the federated view and the job's region for recent local truth. Keep run, execution identity, cluster and physical engine/rank until after delta calculations. Missing rows or a forwarding delay are not idle hardware. [FineLog access and query rules](https://github.com/marin-community/marin/blob/main/lib/finelog/OPS.md).

| Signal / source | Unit and window | Confirmation / contrary evidence |
| --- | --- | --- |
| `async/performance/consumed_loss_tokens` / `cycle_seconds`, step tracker `training_metric_value` | Accepted masked response tokens divided by complete ordinary wall seconds, 2+ warm cycles, then all fleet GPUs | More useful work per GPU-hour at acceptable quality confirms the fleet choice; generated tokens alone do not |
| `wait_for_generation_buffer`, `run_training`, `fwd_logprobs_values_reward`, `sync_weights` phase windows | Seconds and fraction of the same ordinary window | Identifies what to inspect. Overlapping timers do not add to the cycle; use exclusive walls on current code |
| `rollout_groups`, `rollout_group_tokens` by disposition; `rollout_queue_depth` / capacity | Complete groups/tokens and current groups; per update and 5 min | More consumed work with fewer stale/fully-masked groups supports supply tuning; abandoned work raises GPU cost |
| `rollout_staleness_steps` / `consumed_staleness` | Learner step minus generating weight version, per consumed group/token and update | Age within the approved bound supports admission safety; compare quality and mismatch across ages |
| `policy/mismatch/staleness0/*`, pooled absolute log-ratio tails; `policy/ppo_clip_ratio` | Same-token `log p_trainer - log p_vLLM`, quantiles and clipped token fraction per update | Age-zero isolates numerical mismatch; pooled values also contain policy drift. Growth after tuning contradicts numerical safety |
| `cuda_memory_observation`; Iris task cgroup memory | Allocated/reserved/free bytes around each policy phase; host MiB, GiB = MiB/1,024; peak per rank/node | Headroom through publication/backward confirms fit; NVML minus PyTorch reserved can expose non-PyTorch memory |
| Forward/backward scheduler, optimizer, communication/barrier rank spans | CPU dispatch wall seconds per warm update; corroborate with GPU trace when needed | Large waits suggest imbalance/communication; dispatch spans overlap and cannot prove kernel compute utilization |
| Reward, stop reason, length limits, retained evaluations | Complete-answer counts, truncation fraction and held-out score at fixed evaluation settings | Useful answers maintained across seeds support a change; raw reward alone does not establish quality |

<details>
<summary>A bounded FineLog server query to join with learner waits</summary>

This exact historical window is Hero GB200 step45: 408.02 seconds waiting. It showed running requests declining, near-zero queues and low KV, weakening a saturated-server explanation. Change the run/time bounds after discovering your schema; replace the deployment with the run's region.

```sql
SELECT date_bin(INTERVAL '1 minute', to_timestamp_millis(timestamp_ms)) AS minute_utc,
       name, count(*) AS samples, avg(value) AS mean_per_engine,
       max(value) AS peak_per_engine
FROM "telemetry_v1.marinskyrl"
WHERE run_id = 'hero-learning-gb200-01a0dca4-a3'
  AND timestamp_ms >= 1790526804761 AND timestamp_ms < 1790527212781
  AND name IN ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc')
GROUP BY minute_utc, name
ORDER BY minute_utc, name
```

For native vLLM cumulative token counters, take `LAG(value)` within the full series identity, discard negative reset deltas, then sum deltas divided by elapsed seconds. Scan one sample before the window. Native Rigging `work_completed` rows are already deltas; sum them directly. The gauge query above intentionally averages only for a fleet overview: inspect each engine's maximum before ruling out skew or KV pressure.

</details>

## Worked task: short Snowball math RL on H100

Task: a short, single-turn GSM8K run, eight responses per group, throughput per all GPU-hours as the goal, with fixed-token alignment and completed-answer checks. [Historical 26 September source](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5848726769) supports evaluating **P32/I8**, not a universal default.

1. Use the identified Snowball Grug-67B-A2B BF16 export. The measured learner had four eight-H100 nodes, TP1/PP2/EP8/CP1, packed samples, equal microbatches of one, AdamW at 1e-6, frozen query bias and optimizer offload during rollouts. The server had one eight-H100 node, TP1/PP1/DP8/EP8, non-eager batch-invariant vLLM, cap1,024/4,096, memory utilization 0.90 and prefix cache off.
2. Match the task shape: at most 256 prompt + 3,840 response tokens = 4,096 total, 256 prompt groups × eight responses per update. The historical admission settings were 512 generation workers, 256 buffered groups and staleness one. **These are old `trainer.fully_async` fields.** Current code uses `trainer.rollout_buffer` with `max_staleness_steps`, `batch_policy` and `max_in_flight`; it has no direct worker/buffer-field equivalent. Use the current launch plan, inspect the dry run and remeasure supply. A migration is not proven performance equivalence.
3. Check warm cycles two and three, excluding cold first update and final evaluation. The report counts **1,802,197 accepted loss-masked response tokens / 313.7906 s / 40 GPUs = 143.58 tok/GPU-s**, about 516,898 tok/GPU-hour. Wait is 0.081 and 0.684 s, per-engine median running 7.5 and 7.0, median queues zero, KV preemptions zero. Keep I8 for this window and inspect learner/scoring/publication.
4. If your new run has sustained learner wait **and** busy queued/preempting servers, reconsider a server cap or I16 and compare accepted work per all GPU-hours. If engines are idle during the wait, inspect group tails/admission/pauses first. Earlier historical I16 bought 5.8% wall time at 17% more task GPU-hours. More servers need a latency/deadline reason or a measured efficiency gain. [Earlier allocation study](https://github.com/marin-community/marin/issues/8936#issuecomment-5581680479).

The exact job was `/romain/snowball-integrated-h100-01a0dca4-a3`, MarinSkyRL source `0d80999cda9258c7094c720ab66db59f454ee3cc`, vLLM `25f0fc1aae71cccb6da65046cff13145512a6946`. Its resolved config and per-step counters are in `s3://marin-us-east-02a/marin/users/romain/hero-learning-01a0dca4/snowball-integrated-h100-a3/report.json`. The finite warm sample and changed runtime limit generalization. Total short-run cost must also include the evaluations and startup.

## Performance and numerical alignment

Hero's accepted fixed-weight rule is fewer than 0.1% same-token trainer/serving probability ratios outside [0.8, 1.2]. Replay must use captured routes for the same tokens, positions, masks and weights. A native-route/sentinel performance bank does not qualify numerical agreement or learning. [Numerical decision](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5835032707).

| Choice | Measured result | Starting decision / remaining gap |
| --- | --- | --- |
| Captured versus native expert routes | H100 5 versus 114 outliers / 50,962 tokens; native fails the gate. Replay scoring 5.147 versus 4.146 s when native replay controller is removed | Keep replay. Controller removal is an additional change; this is scoring cost, not whole-RL cost. [Matched study](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5848726769) |
| Learner batch invariance OFF | H100 fixed-bank warm update 18.538→18.473 s; both captured-route gates pass | Keep ON for the qualified online path; safety/accepted throughput of OFF is unknown. Same source study |
| GB200 learner fused→flash | Warm 63.303→61.180 s (3.4%); same-token replay gates both pass | A learner-only option; online safety and accepted throughput unmeasured. [Evidence](#learner-evidence) |
| Publication NCCL bundle OFF | GB20048 fixed bank: ON costs 6.7% learner time; OFF has no safe online publication result. H100 OFF failed at 96 and 128 GPUs | Keep ON on tested H100 layouts. Matched 128-GPU OFF step0 used 16.59 GiB more non-PyTorch memory; no warm OFF rate. [Evidence](#learner-evidence) |
| S=1→S=2 GB200 admission package | Workers 32→48 and buffer16→32 also change; accepted rate 75.46→89.56 tok/s over 56 GPUs | Short combined-setting result, not a staleness-only or long-run learning claim. [Online evidence](#online-evidence) |
| GB200 broadcast→expert_block | 89.56→153.05 accepted tok/s; four-cycle publication 737.93→117.27 s. Fixed-weight, sampled readback and separate full byte replay passed | Trial revisions differ from draft #860's combined branch; GPU validation there is still required. Responses differ by 5.44%; +70.9% is observed, not a universal transfer speedup. [Online evidence](#online-evidence) |

Recomputation reduces activation memory by repeating compute. Optimizer offload trades device headroom for host memory and transfer time. Microbatch changes can alter kernels and numerical results; keep forward/train microbatches equal. Check memory at publication as well as training, keep checkpoint geometry compatible, and recheck fixed-token gap plus online clipping before accepting a throughput win. [Current Grug mechanism](grug-megatron-training.md).

### Learner evidence

<details>
<summary>Fixed-bank reports and qualification limits</summary>

The bank SHA256 is `6d97026e4235babacdfe1d04d4e53d9e7deef4c68e1c96a6be73a12ced09c896`. Prefix **H** = `s3://marin-us-east-02a/marin/users/romain/hero-learning-01a0dca4/`; **P** = `s3://hero-checkpoints/marin/users/romain/hero-perf-01a0e406/`. Append the report key shown. These locators require authorized CoreWeave access. Each report contains the resolved config, source and input digests.

| Point | Iris job; source SHA | Report key; SHA256 |
| --- | --- | --- |
| Snowball H10032 | `/romain/snowball-learner-h100-32g-01a0dca4-a1`; `fa7059545a8a6ec62456fda3bdb9ce05dae103f8` | H `snowball-learner-h100-32g-a1/report.json`; `6f0e7faac71c545eb58395846b316b7fe7b988ce007b05e10271227c0fbb5b74` |
| Hero H10096 | `/romain/hero-learner-h100-96g-01a0dca4-a2`; `80907a6de65c49cee29ca3b13c2fee27d2a05556` | H `hero-learner-h100-96g-a2/report.json`; `7d20654dda9bcb4435459e89a895806e89958de778aa4a3dd783e60bf9bf1e73` |
| Hero GB20048/64 | `/romain/hero-gb200-learner-48g-01a0e406-a1` / `64g` counterpart; `ca91727c765e6f79399b8844b195108cc78a6ee1` | P `gb200-learner-48g-a1/report.json` / `gb200-learner-64g-a1/report.json`; hashes `14078f3e73dc17ab109a717df74fe0af769a39dfd137ef4e8f06a1597fcd611b` / `5fc6aa81919460a21e4576e00abdc8aa71500ca965a6545913eba2518260ead8` |
| Snowball GB20016/32 | `/romain/gb200-snowball-learner-16g-01a0e406-a1` / `32g` counterpart; same GB200 source | P `gb200-snowball-learner-16g-a1/report.json` / `gb200-snowball-learner-32g-a1/report.json`; hashes `dc5ad08a13cdd37b9a9982a3c45d56cb8696edf9e52fc450e126a43a3eccd63f` / `d926dea95c1b497e58161c91f4e498361a78b451c3b1fe474ca04f2fda2a774f` |
| GB200 fused/flash | `/romain/hero-gb200-tim-fused-48g-01a0e406-a2` / `/romain/hero-gb200-tim-flash-48g-01a0e406-a1`; same fixed bank | P `gb200-tim-fused-48g-a2/report.json`, `gb200-tim-flash-48g-a1/report.json`; SHA256 `fc76835736d86baacbcc1f8bfc0d38c973eb4bc72db0dbb2b5e362bbecfeca51` / `a482ca6521a353cd511dd99cce4721850f52d6c5204041dcf48a95c9b3a64405` |
| GB200 NCCL ON/OFF | `/romain/hero-gb200-nccl-on-48g-01a0e406-a2` / `/romain/hero-gb200-nccl-off-48g-01a0e406-a1`; source `1bc2affa3d54da08285faed2b2750785082f3272` | P `gb200-nccl-on-48g-a2/report.json`, `gb200-nccl-off-48g-a1/report.json`; SHA256 `d87a98d01940c2b4091c5c089756e604c093efda7b3879d2fa3450c993aa735c` / `6cf46d33907e72c5977b7945156abce01e7341c8ea042e6693c7f00144352956` |
| H100128 NCCL ON/OFF | same 16 nodes; OFF failed backward after step0 | H100 performance prefix `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/`, keys `h100-nccl-on-128g-a1/report.json` / `h100-nccl-off-128g-a1/report.json` |

Hero fixed-bank results use MuonH/AdamH/Adam; Snowball uses AdamW. Warm training excludes scoring, optimizer offload outside the timer, serving and publication. Reported rank dispatch phases overlap. Results from one placement and three warm updates do not establish fleet minima or variability.

The NCCL bundle was tested together: `NCCL_LAUNCH_MODE=GROUP`, `NCCL_COLLNET_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `NCCL_P2P_NET_DISABLE=1`, `NCCL_MIN_NCHANNELS=1`, `NCCL_MAX_NCHANNELS=1`, `NCCL_PROTO=Simple`, `NCCL_ALGO=allreduce:tree`, `NCCL_NTHREADS=1`, `NCCL_SOCKET_NTHREADS=1`. The experiment's ON/OFF switch is not a production launcher argument or a per-flag result.

</details>

### Online evidence

<details>
<summary>Complete-cycle measurements and current integration status</summary>

| Fleet | Warm accepted response work / complete cycle wall | Bound and durable source |
| --- | --- | --- |
| Hero H100128+32, steps2–5 | 8,859,198 / 4,975.25 s = 1,780.65 tok/s, 11.13 per all GPUs | `/romain/hero-integrated-h100-01a0dca4-a8`, source `3d455ebf648fb3a58764f20afa5923a7afffa0c5`; H prefix `h100-integrated-batch512-noprof-a8/report.json`. Batch512×8, staleness1, cap64/512. Wait14.9%, full training53.1%, publication18.2% |
| Hero GB20048+16, 29 ordinary updates | 1,429,495 / 19,419.66 s = 73.61 tok/s | `/romain/hero-learning-gb200-01a0dca4-a3`, source `d9c9997ea13c314fd2ddd2c698b8d2d50320e2a0`; `s3://hero-checkpoints/marin/users/romain/hero-learning-01a0dca4/gb200-learning-a3/report.json`; SHA256 `e7e6e7cc6b8d8938e21cd6700f426ee7ca1815d6a864151dcc8aed04277c079a`. Batch16×8; wait57.9%, publication28.3% |
| Hero GB20048+8 S=2 cap16, steps34–37 | 161,621 / 1,804.63 s = 89.56 tok/s, 1.599 per all GPUs | `/romain/hero-gb200-feed-48p8s-s2-01a0e406-a1`, source `8d71544751a41d978bb476a0c49e9e30af21921a`; P prefix `gb200-feed-48p8s-s2-a1/report.json`. Batch16×8, workers48, buffer32. Maximum clip0.02441%, age2; wait40.6%, publication40.9% |
| Same fleet, expert_block | 170,418 / 1,113.45 s = 153.05 tok/s | `/romain/hero-gb200-expert-block-48p8s-s2-01a0e406-a4`, source `e26957594e84c67c88888b7c0653100905fb3d77`; P `gb200-expert-block-48p8s-s2-a4/report.json`; SHA256 `d3eab5bdde398985bb49f5c1775c8c096debd71d25fd10e9bb2c1e776884d346`. Max clip0.02955%; sampled weights exact |

P and H are the prefixes in learner evidence. A4's written report passed five updates and final evaluation; its deleted head pod required manual Iris completion. Separate `/romain/hero-gb200-expert-block-verify-48p8s-s2-01a0e406-a6` succeeded 14/14 and checked full receiver-parameter bytes plus trainer DP replicas at step32, with no updates or evaluation. Source `97bae9fbf285d95caa569a01d17065d0435c34e9`; P `gb200-expert-block-verify-48p8s-s2-a6/report.json`; SHA256 `ca5c3022ecf6bf777a6b0bf66dc3db402e0428bd57350a84337b9f6921fdf65e`. Per-receiver byte counts were not persisted.

The online cap comparisons are `s3://marin-us-east-02a/marin/users/romain/hero-perf-01a0e406/h100-online-128seq-a2/report.json` and P `gb200-feed-48p8s-s2-maxseq32-a3/report.json`. The H100 retry also raised the pause deadline from 120 to 300 s; four warm steps and independently generated outputs limit causal inference.

On 30 September, [Hero architecture #792](https://github.com/marin-community/MarinSkyRL/pull/792) and [vLLM #77](https://github.com/marin-community/vllm/pull/77)/[#78](https://github.com/marin-community/vllm/pull/78) were merged. [Runtime wheels #799](https://github.com/marin-community/MarinSkyRL/pull/799), [expert-block Hero integration #860](https://github.com/marin-community/MarinSkyRL/pull/860), and [sparse publication #701](https://github.com/marin-community/MarinSkyRL/pull/701) remained drafts. A successful trial source or CPU CI does not qualify their combined released runtime.

</details>

## Measurements that could change a recommendation

| Gap | Narrow proposed probe | Cost and decision value |
| --- | --- | --- |
| Current rollout-buffer migration for the Snowball worked task | Same 4K bank, P32/I8, two warm cycles plus fixed-token gate; inspect resolved admission semantics | 40 H100s × roughly 30 min = 20 GPU-hours including an estimated startup allowance. Would validate current supply/accepted-rate advice; measure actual cost |
| Hero expert-block combined branch | Step32 restore, exact full-byte publication replay, then four ordinary matched cycles on P48/I8 | 56 GB200s × roughly 1 hour = 56 GPU-hours including restore/eval allowance. Would decide release readiness and whether publication savings survive integration |
| Hero/Snowball isolated GB2008 serving | Fixed 4K bank at two client loads on a reliably constrained rack; verify placement before loading | 8 GPUs × roughly 30 min = 4 GPU-hours plus startup variance. Would decide whether the 16-GPU isolated server reference can shrink; old failed placements produced no sizing data |
| Hero long-response or multi-turn >4K RL | First qualify fixed-token routes/positions at one intended length, then a bounded complete task with tools/tails | No defensible fixed cost yet: learner fit and runtime are unknown. Serving 256K short outputs and synthetic 65K AdamW updates cannot predict this rate |

These are **estimated probe budgets**, not scheduled work. [Synthetic full-Hero capacity](https://github.com/marin-community/MarinSkyRL/blob/e3f5fff039f9b965d3f014584f8abd34164c0381/docs/grug-megatron-training.md#full-hero-capacity) passed 65K save/restore on 256 H100 or 64 GB200 with AdamW. MuonH long-context learner fit, online quality, sustained decode and 128K/256K trainer rates remain unknown. No new GPU experiment is required to use or review these drafts.
