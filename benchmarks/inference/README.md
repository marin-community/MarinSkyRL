# Inference replay

`snowball.py` runs the production inference client with a frozen curriculum and
stock vLLM. It owns inference engines only. It does not train or estimate RL-step
speedup. Reserve interactive H100 nodes through Marin's `use-iris` workflow before
running it; release the holder after copying results to durable regional storage.

Bootstrap the locked runtime with `cloud/iris/bootstrap_runtime.sh`, then stage
the model and corpus on each participating node:

```bash
python benchmarks/inference/prepare_snowball.py --output /tmp/snowball-input
```

This reads the regional curriculum and thinking-step-630 export. It samples 64
prompt groups with eight repetitions, preserves production session hashing, and
selects batches whose first node receives exactly 128 requests. Selection uses
only group IDs, before generation. The corpus retains tokenized inputs and its
source hash. Reuse the same `corpus.json` for every treatment and node.

Run through the Iris telemetry wrapper so the existing Finelog sink receives
native metrics. Use a unique run ID, result directory, and command directory:

```bash
export SKYRL_RUN_ID=snowball-example
python -m cloud.iris.telemetry_env -- python benchmarks/inference/snowball.py \
  --model /tmp/snowball-model --corpus /tmp/snowball-input/corpus.json \
  --output /tmp/snowball-results --commands /tmp/snowball-commands \
  --source-sha "$(git rev-parse HEAD)"
```

For a four-node confirmation, start a private Ray cluster on the four reserved
nodes, export its `RAY_ADDRESS`, and add `--nodes 4`. Each node needs the same
runtime, source, model path, and corpus. The driver records the effective config,
source/lock hashes, package versions, and producer identities.

After `ready.json` appears, atomically move a JSON command into the command
directory. Commands run in filename order and become `.done` files. For example:

```json
{"name":"burst-a","mode":"burst","concurrency":128,"waves":[1,2]}
```

Use `"mode":"refill"` with the same fields to replace each completed request
immediately. A burst waits for every request in each concurrency-sized wave.
Waves are zero-based corpus indices and may repeat. On four nodes, use 512
outstanding requests for the production-sized comparison. Each treatment clears
the prefix cache; reuse within the treatment remains enabled. Warm the engines
before collecting clean comparisons. Sampling and token budgets stay fixed at
1024 prompt, 8192 output, and 9216 total tokens, with natural stopping.

Optional `profiles` entries contain `name`, `after_seconds`, and `seconds`.
They invoke the stock PyTorch profiler on every physical inference engine. Keep
these commands separate from clean measurements; profile export can extend the
treatment. Use a fresh engine process for each capture: restarting the profiler
with the locked stock wheel produced a CUDA launch failure during live validation.
A command containing `{"stop":true}` stops the driver and tears down
its engines. The driver also stops accepting commands after `--max-seconds`.

`events.jsonl` retains full per-engine snapshots, each engine's poll start/end
times, before/after baselines, treatment boundaries, and compact request receipts.
It omits generated text and token dumps. Native request UUIDs appear only in the
artifact, never as Finelog labels. First-token time is when the async vLLM wrapper
receives its first nonempty output, not a GPU kernel timestamp. Ordinary runs
retain no request timings unless `generator.capture_request_timings=true`.

```bash
python benchmarks/inference/analyze.py /tmp/snowball-results
```

The analysis writes JSON and per-engine/aggregate CSV. It checks producer and
histogram completeness, reconciles counters with receipts, and uses each engine's
actual monotonic collection interval. Whole-treatment rates include the final
drain. Refill steady rates use complete poll intervals from ten seconds after
startup through the last request submission; the summary reports how many seconds
qualify. Profiled arms remain marked in the output. Preserve raw events and
profiles alongside the summary so these choices can be checked independently.
