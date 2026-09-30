# Choose a Snowball or Hero performance path

Use these pages to choose a BF16 starting layout on CoreWeave H100 or GB200/B200, check what limits the task, and make one adjustment. Evidence was checked on 30 September 2026. A historical layout is a starting example; reproduce its workload and numerical contract before treating its rate as a target.

| Your task | Write down | Starting path | First check |
| --- | --- | --- | --- |
| Interactive serving | Prompt/output length distribution, burst concurrency, p95 first-token and response-time limits | [vLLM, standalone serving](performance-vllm.md#standalone-serving) | Queue time and p95 latency within your limit |
| Bulk inference | Input/output lengths, client concurrency, deadline, acceptable output behavior | [vLLM, supplied throughput examples](performance-vllm.md#measured-serving-examples) | Useful completed output tokens per GPU-hour, with errors and long tails |
| Short single-turn RL | Responses per group, groups per update, response cap, permitted policy age | [Megatron and the full loop](performance-megatron.md#choose-a-learner), then its [Snowball H100 example](performance-megatron.md#worked-task-short-snowball-math-rl-on-h100) | Learner wait together with running/waiting requests |
| Multi-turn coding RL | Turns, generated tokens, tool time/text, peak accumulated context, group completion tails | [Full-loop diagnosis](performance-megatron.md#find-the-full-loop-limit) and [vLLM context limits](performance-vllm.md#context-and-qualification) | Completed usable groups and time in tools versus model calls |

Also record the model/export revision, accelerator and GPU count, combined context cap, and your objective. For latency, name a time limit. For GPU efficiency, name the useful work and deadline. For RL, numerical alignment and usable answers constrain every performance choice.

**Hero's qualified learning evidence covers non-agentic GSM8K at 4,096 total tokens.** It uses TP1, non-eager batch-invariant vLLM, recorded expert-route replay, frozen query bias, and fresh MuonH/AdamH/Adam state (or its full RL restore). Longer-context serving probes establish fit and short-output performance only. Multi-turn Hero RL remains provisional. The later staleness-two supply trial is a short performance result; the 64-update learning curves used staleness at most one. See the [qualification and learning audit](https://github.com/marin-community/MarinSkyRL/issues/737#issuecomment-5860256429).

## Estimate an unfamiliar task cheaply

Start with 16–32 existing representative prompts or trajectories, including the longest known cases. Use the exact tokenizer and chat template of the chosen checkpoint. This sample is a rough planning estimate, not a reliable p99. Reuse an existing short smoke when generated lengths are unknown; a new GPU sweep is unnecessary for the first draft configuration.

| Shape | What to count | How it changes the choice |
| --- | --- | --- |
| Input | Tokenized prompt at each model call, including system text and tool results | Large prefills compete with decode and raise first-token latency |
| Output | Generated tokens per response segment and whole trajectory; median, sample maximum, and truncation count | Long outputs retain KV and delay the last member of a group |
| Context | Largest input plus generated continuation at a call | Must fit model, rotary, engine and trainer bounds; tool text consumes context |
| Concurrency | Simultaneous model calls; for bursts, requests arriving within one response duration | Choose client and scheduler caps; tool-bound tasks may keep few calls active |
| Groups | Responses per prompt and complete groups per update | Eight responses per group means a 16-group update needs 128 trajectories; the slowest response can hold the group |
| Tool work | Tool wall seconds, text tokens returned, failed or timed-out tools | More serving GPUs will not remove tool waits |

For a rough supply budget, estimate `groups needed per second × responses per group × mean generated tokens`. Compare that demand with a measured server rate at a similar length and concurrency. Label the result **estimated**: admission limits, pauses, scoring, rejected groups and tails can prevent that work reaching the learner. Check actual running requests before increasing workers or servers.

## Keep the three rates separate

| Rate | Numerator | Denominator and use |
| --- | --- | --- |
| Learner-only | Unpadded prompt + response sequence tokens | Warm training seconds × learner GPUs; choose a learner layout |
| Supplied serving | Native generated decode tokens | A supplied measurement window × serving GPUs; inspect server capacity |
| Whole RL | Accepted, loss-masked response tokens | Complete ordinary cycle seconds × all learner and serving GPUs; choose the fleet |

Multiply tokens/GPU-second by 3,600 to obtain tokens/GPU-hour. For a short job, also report total useful work divided by task-running GPU-hours including setup, evaluation, saving and teardown. Reservation and billing time are separate. A faster fixed-update run can consume more tokens or produce worse completed answers. [Snowball's historical async study](https://github.com/marin-community/marin/issues/8936#issuecomment-5581680479) demonstrates both costs.

## End with one starting choice

For short Snowball H100 math RL, start by evaluating the historical **32 learner + 8 serving GPU** example and its [exact shape and evidence](performance-megatron.md#worked-task-short-snowball-math-rl-on-h100). Check warm learner wait and server activity together. If wait is already negligible, retain the serving allocation and inspect training/scoring/publication. If wait grows with high queues or preemptions, use the [vLLM diagnosis table](performance-vllm.md#read-the-signals-together).

**This is a sizing candidate, not a currently runnable Marin recipe.** Marin's normal Snowball entrypoint fixes 128 prompts × four answers and still emits removed async settings. Its recipe and runtime translation need an update before a dry run can validate the worked 256 × eight task. See the [current launcher gap](performance-megatron.md#worked-task-short-snowball-math-rl-on-h100).

For other tasks, select the closest measured row in the two guides. Cells without matching evidence are **unknown**. Keep a proposed probe smaller than the work it might save: state the decision, use a fixed small bank, and include startup cost before allocating GPUs.
