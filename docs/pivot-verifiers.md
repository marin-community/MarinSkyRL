# Pivot dataset verifiers

The four public NVIDIA Pivot datasets can be graded on CPU in MarinSkyRL. The
verifiers are implemented locally; they do not import or execute NeMo Gym code.
`NemotronUltraEnv` uses the same graders during single-action training rollouts.

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
uses 32 learner GPUs at TP1/PP2/CP1/EP8 and 32 rollout GPUs. Sample packing is
disabled. CP1 keeps each sequence on one rank and avoids the pinned Transformer
Engine 2.11 restrictions on context parallelism across multiple ranks:

- p2p rejects sliding-window attention.
- All-gather rejects packed sequences, which the trainer requires for CP > 1.
- All-to-all CP2 requires an even split of KV heads; the smoke model has five.

These restrictions are enforced in
[Transformer Engine's context-parallel attention](https://github.com/NVIDIA/TransformerEngine/blob/v2.11/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py).

Run the launcher without `--run` to validate the recipe locally:

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
