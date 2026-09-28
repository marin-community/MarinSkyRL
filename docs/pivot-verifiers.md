# Pivot dataset verifiers

The four public NVIDIA Pivot datasets can be graded on CPU in MarinSkyRL. The
verifiers are implemented locally; they do not import or execute NeMo Gym code.
`NemotronUltraEnv` uses the same graders during single-action training rollouts
for rows prepared with explicit `extra_info.nemotron_ultra.pivot_dataset` metadata.
Other Ultra rows retain exact argument matching and reject extra tool calls.

## Reference behavior

The reference is [NeMo Gym commit
1c8261080bdc881b3e9b7f870e6418f160516991](https://github.com/NVIDIA-NeMo/Gym/tree/1c8261080bdc881b3e9b7f870e6418f160516991),
the `main` head inspected for this implementation. Dataset profiles follow its
[single-step tool comparison configurations](https://github.com/NVIDIA-NeMo/Gym/tree/1c8261080bdc881b3e9b7f870e6418f160516991/resources_servers/single_step_tool_use_with_argument_comparison/configs)
and [Terminal string-only configuration](https://github.com/NVIDIA-NeMo/Gym/blob/1c8261080bdc881b3e9b7f870e6418f160516991/resources_servers/terminus_judge/configs/terminus_judge_string_only.yaml).

| CLI dataset | Hugging Face dataset under `nvidia/` | Verifier |
| --- | --- | --- |
| `function_calling` | `Nemotron-RL-Agentic-Function-Calling-Pivot-v1` | Action comparison, word threshold 0.1 |
| `swe` | `Nemotron-RL-Agentic-SWE-Pivot-v1` | Action comparison, word threshold 0.0 |
| `terminal` | `Nemotron-RL-Agentic-Terminal-Pivot-v1` | Terminus schema, completion, command similarity >= 0.9 |
| `conversational_tool_use` | `Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1` | Action comparison, word threshold 0.1 |

### Action comparison

Function names must match. Arguments are decoded as JSON and compared recursively:
object keys and list lengths must match; list order matters; float differences
must be strictly below `1e-6`. Type checks follow Python `isinstance`, including
the reference's integer/boolean asymmetry.

For strings with at least two words on both sides, the score is the multiset word
overlap divided by the sum of both word counts. Words are lowercased. Identical
strings score **0.5**, not 1. Strings with fewer than two words on either side
require exact original equality. SWE's zero threshold therefore still checks
structure, numbers, and short strings.

Tool calls take precedence over accompanying text. A reference message accepts
any assistant text, including an empty string. Batches need distinct matches
for every expected call, in any order. Extra generated calls are allowed by the
published default configuration. Malformed candidate arguments fail to match;
malformed reference arguments raise a data error. Local diagnostic categories
are intended for inspection, not byte-for-byte upstream response compatibility.

### Terminal

Both reference and candidate must be JSON objects satisfying the row's
`metadata.harness` schema (`terminus_1` or `terminus_2`). Extra properties are
rejected. A true reference completion flag requires the candidate flag to be
true; a false reference flag imposes no completion constraint.

After removing text through the last `</think>`, the verifier compares ordered
command `keystrokes` concatenated without a separator using Python
`difflib.SequenceMatcher` with its default settings. Analysis, plans, and duration
values must satisfy the schema but do not enter the similarity score. A row's
`threshold` overrides 0.9. Empty command lists match each other; an empty list
does not match a list containing an empty command string. Markdown JSON fences
are not stripped.

These are local action rewards. They do not execute tools, run repository tests,
launch terminal tasks, or call an LLM judge. The SWE profile follows the released
dataset's argument comparator; the separate NeMo Gym `swe_pivot` resource server
is a different verifier. This implementation does not establish reproduction of
the paper's profiling, sampling, or training results.

## Run locally

Run from the MarinSkyRL root. With an existing environment, `--no-sync` avoids
resolving unrelated trainer dependencies. `PYTHONPATH` selects this checkout's
Gym code even if the virtualenv contains an older installed package.

```bash
# If an environment has not been installed yet:
uv sync --frozen --group dev --group harbor-test --extra cpu --extra telemetry

# Download only a prefix of each release, then check its reference responses.
for dataset in function_calling swe terminal conversational_tool_use; do
  PYTHONPATH=skyrl-gym:. uv run --no-sync -m infra.rl_data.pivot sample \
    --dataset "$dataset" --limit 16 --output "/tmp/pivot/$dataset.jsonl"
  PYTHONPATH=skyrl-gym:. uv run --no-sync -m infra.rl_data.pivot replay \
    --dataset "$dataset" --limit 16 --input "/tmp/pivot/$dataset.jsonl"
done
```

Sampling streams at most 32 MiB per invocation and closes the response after the
requested rows. A `.manifest.json` sidecar records the resolved Hugging Face
commit and filename. Supply `--revision COMMIT` to repeat a sample. For existing
local JSONL files, use `replay`, `grade`, or `prepare` directly; they need no network.
The row limit defaults to 16 for every command.

`replay` checks that reference responses score 1 and missing responses score 0;
it exits nonzero on failures. This checks dataset wiring, not model quality or
independent numerical parity with a running upstream verifier.

### Grade model predictions

Predictions are JSONL, with one entry per selected input row and consecutive
zero-based indices. `response` accepts either a Chat Completions assistant
message or a Responses API object containing `output`. Tool calls must be
structured: apply the model's tool parser before grading textual tool markup.

```json
{"index":0,"response":{"role":"assistant","content":null,"tool_calls":[{"type":"function","function":{"name":"search","arguments":"{\"query\":\"red blue\"}"}}]}}
```

For Terminal, place the generated JSON **string** in `response.content`.

```bash
PYTHONPATH=skyrl-gym:. uv run --no-sync -m infra.rl_data.pivot grade \
  --dataset function_calling --input /tmp/pivot/function_calling.jsonl \
  --predictions /tmp/predictions.jsonl --limit 16 --output /tmp/rewards.jsonl
```

The command writes per-row rewards and diagnostics, then prints mean reward and
category counts. Count or index mismatches raise an error.

### Prepare training rows

```bash
PYTHONPATH=skyrl-gym:. uv run --no-sync -m infra.rl_data.pivot prepare \
  --dataset terminal --input /tmp/pivot/terminal.jsonl \
  --limit 16 --output /tmp/terminal.parquet
```

Prepared rows use `env_class=nemotron_ultra`, preserve request tools and generation
options, and carry the reference record as JSON. Responses function calls and
outputs retain their call IDs; reasoning items are omitted from chat history.
`extra_info.trajectory_id` preserves the source grouping key: `trajectory_id`
for action datasets and `metadata.source_trajectory_uid` for Terminal. Split by
this field before creating training and evaluation sets. The preparation command
does not create a split or filter for prompt length.

## Grug smoke preflight

The SWE smoke for `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`
uses 32 learner GPUs at TP1/PP2/CP1/EP16 and 32 rollout GPUs. Sample packing is
enabled. CP1 keeps each sequence on one rank and avoids the pinned Transformer
Engine 2.11 restrictions on context parallelism across multiple ranks:

- p2p rejects sliding-window attention.
- All-gather rejects packed sequences, which the trainer requires for CP > 1.
- All-to-all CP2 requires an even split of KV heads; the smoke model has five.

These restrictions are enforced in
[Transformer Engine's context-parallel attention](https://github.com/NVIDIA/TransformerEngine/blob/v2.11/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py).

The policy and reference log-probability paths use 128-token chunks. The
64-GPU smoke failed during the first policy backward pass with a 1024-token
chunk: the policy and colocated reference processes left less than 1 GiB free
on an H100. A subsequent 128-token run exhausted memory allocating the full
logits gradient. EP16 shards expert weights across twice as many ranks while
retaining the same 64-GPU allocation. The smoke logs to `marin-community/pivot-rl`.
Set `WANDB_API_KEY` in the launch environment; the launcher forwards it to
the GPU job.

With `WANDB_API_KEY` set, run the launcher without `--run` to validate the recipe locally:

```bash
PYTHONPATH=skyrl-train:skyrl-gym:. uv run --no-sync skyrl-train/ci/pivot_grug_smoke.py \
  --run-id grug-preflight \
  --output-root /tmp/grug-preflight/output \
  --temporary-root /tmp/grug-preflight/temporary
```

This checks the launch topology, trainer configuration, and this smoke's CP1
requirement before dataset preparation, model caching, or Iris submission.
It runs on CPU and does not load model weights. Passing does not establish CUDA
kernel compatibility or sufficient GPU memory.

Use `--steps 10 --train-prefixes 512 --compare-sft` for the paired experiment.
The coordinator prepares one dataset, then runs RL and reference-action SFT
sequentially from the same pinned Grug checkpoint. Both arms use the same 512
training prefixes, 128 held-out tasks, 10 updates, learning rate, and greedy
evaluation at steps 0 and 10. Training and probes have disjoint
trajectory and task IDs. Reference actions must fit the 512-token completion
budget; this selects a bounded subset of the released dataset.

Each update consumes all 512 training prefixes. RL samples 16 completions per
prefix; SFT supervises one released action per prefix with token cross-entropy
and no KL penalty. The arms match prefixes and updates, with different
completion-token and compute budgets. SFT evaluation generates answers
through the same inference engines and verifier as RL. Per-arm reports live
under `rl/diagnostics` and `sft/diagnostics`; `diagnostics/comparison.jsonl`
pairs their predictions and `diagnostics/summary.json` reports mean rewards
and paired wins. The trainer masks generation server errors using its standard
error handling. The paired comparison rejects evaluation responses with errors;
training errors remain visible in the retained action metrics.

Generation uses 64 concurrent sequences per engine and the frozen EAGLE-3
draft `laion/snowball-64k-eagle3-draft-r2egym` at revision
`4bdb47c08e5b5190bea3c7a93c3e14470230e469`, with three speculative tokens.
The 32K request window, batch size, packing, and serving configuration match
[the rollout-buffer experiment](https://echo.oa.dev/wiki/541). This comparison
retains the Sept 21 model, synchronous trainer, GRPO objective, and Pivot SWE
data; it does not adopt the training-loop refactor in PR #774.

Both decoding configurations explicitly stop on Grug's end-of-text token
128001 and end-of-turn token 128009. The latter terminates assistant messages
in the pinned chat template and must also be recognized by constrained decoding.

For a smaller GPU check, the existing
`skyrl-train/tests/gpu/test_grug_megatron.py::test_grug_megatron_pp2_train_step_updates_weights_and_exports`
test uses two H100s and a tiny Grug checkpoint to exercise a training step and
export. It does not validate the full 64-GPU placement or the 67B model's memory use.

## Tests

```bash
PYTHONPATH=skyrl-gym:. uv run --no-sync pytest \
  skyrl-gym/tests/test_nemotron_ultra.py \
  infra/tests/test_pivot.py infra/tests/test_pivot_swe_smoke.py
```

The tests cover threshold boundaries, recursive arguments, batch matching,
response normalization, Terminal schema and completion checks, prediction-file
alignment, and dataset-to-Parquet-to-Gym round trips. Existing tests in the same
Gym file also exercise other verifiers and may need their local dependencies or
caches.
